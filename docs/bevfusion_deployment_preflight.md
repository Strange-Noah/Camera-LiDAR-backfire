# BEVFusion / TransFusion-L preflight — 2026-10-19

**Prepared; paid GPU authorization pending.** No GPU created or started. Both
existing CPU Pods remain EXITED. No successful model inference is claimed.

## Models

Official source: https://github.com/mit-han-lab/bevfusion (archived; Apache-2.0).
README reference full-validation mAP/NDS: BEVFusion 68.52/71.38;
TransFusion-L 64.68/69.28. These are not mini acceptance targets.

`environment/bevfusion/upstream.json` records the exact two configs, individual
official Dropbox URLs, source revision and download-script revision. Both local
HEAD requests timed out (30s / 20s). Current reachability, byte sizes and SHA256
are **unverified**. Download only `bevfusion-det.pth` and `lidar-only-det.pth`
into `/workspace/checkpoints/bevfusion/`; the downloader records their hashes.
Check download access from the Pod before spending time on compilation.

Current upstream HEAD adds Radar to the shared detection pipeline. The environment
pins the official pre-Radar revision `db75150717a9462cb60241e36ba28d65f6908607`.
No source patch is applied. Strict checkpoint matching remains a runtime gate.
All runtime config adjustments are documented in the environment README.

## GPU decision

Live Runpod evidence: `bevfusion_inventory.json`. Prices are Secure catalog rates,
not a reserved offer or a created Pod's actual charge. Only EU-RO-1 stock can use
the existing 50 GB STANDARD volume `p6vb29fmlu` without migration.

| GPU | VRAM GB | USD/h | EU-RO-1 stock | Decision |
|---|---:|---:|---|---|
| RTX 3090 | 24 | 0.50 | Not listed | Ampere, unavailable here |
| RTX 4090 | 24 | 0.89 | MEDIUM | Recommended mini compatibility trial |
| RTX A6000 | 48 | 0.59 | Not listed | Preferred if stock returns |
| A40 | 48 | 0.59 | Not listed | Preferred if stock returns |
| A100 PCIe/SXM | 80 | 1.79 | Not listed | Unavailable here |
| L4 | 24 | 0.59 | LOW | Cheaper-hourly Ada alternative |
| RTX PRO 4000 Blackwell | 24 | 0.57 | LOW | Exclude from legacy-stack trial |
| RTX PRO 4500 Blackwell | 32 | 0.72 | MEDIUM | Exclude from legacy-stack trial |

Recommended GPU: one RTX 4090 (provisional until acceptance)

GPU VRAM: 24 GB

Datacenter: EU-RO-1, Secure Cloud, existing volume mounted at /workspace

Price per hour: USD 0.89 compute; recheck before creation

Estimated deployment cost: planning allowance 2–4 GPU hours = USD 1.78–3.56
compute, plus container disk. Proposed ceiling: USD 5 incremental spend and at
most four GPU hours, excluding existing volume rent. This is not a completion
guarantee. Preserve outputs and release the new Pod at completion or the cap.

Expected inference memory: **unknown until measured**; official docs do not
establish the peak for this setup. Batch 1, sequential model processes, official
FP16, initially zero loader workers. 24 GB is trial capacity, not proven adequate.
Do not reduce resolution/sweeps to conceal OOM. Request at least 32 GB host RAM,
8 vCPU, 16 GB shared memory and 60 GB disposable disk; verify actual allocation.

CUDA compatibility: CUDA 11.3.1/PyTorch 1.10.1 cu113 and upstream sm_86 cubins
have an architecture-compatible path to Ada sm_89. Actual dependency and model
kernels must still pass. Blackwell lacks this same-major cubin compatibility.
Runpod host CUDA versions are not the container toolkit version.

Alternative GPU: L4 24 GB at USD 0.59/h; A40/A6000 only when EU-RO-1 stock returns.
4090 costs 1.51x L4 per hour and only wins on inference cost if its throughput
exceeds L4 by that ratio. No measured speed or cost winner is claimed.
Do not migrate data or change DC without authorization.

Compatibility source: https://docs.nvidia.com/cuda/ada-compatibility-guide/ .

## Deployment and acceptance

| Stage | Status | Missing proof |
|---|---|---|
| Environment | Files prepared | CUDA build, package resolution, ops/kernel checks |
| Model loading | Not run | Both strict loads, checkpoint manifests |
| Same sample | Not run | Matching tokens, boxes/classes/scores, visualization review |
| Mini smoke | Not run | Success counts, device memory, RSS, timings and errors |
| Full baseline | Planning only | Complete validation data and cost approval |

Local Python/shell syntax checks do not establish CUDA correctness. This Mac has
no Docker builder or target GPU. Runtime pins are candidates, not an accepted
complete lock. The Dockerfile and bootstrap setup are not yet build-tested.
Capture actual image digest, OS packages, pip freeze, nvidia-smi and build logs
after deployment, and verify rebuild before claiming reproducibility.

Prior audits establish mini presence; no live volume filesystem inspection is
possible while both attached CPU Pods are stopped. Recheck actual `du` usage
against the purchased 50 GB quota, not shared-pool `df`, before downloads.
Preserve the dataset permissions and existing Landlock member workflow.
Generate infos using the official converter in a writable symlink view; skip the
training-only GT database. Required inputs are six camera views, current LiDAR
plus nine sweeps, calibration, ego poses, labels and mini infos. Radar and map
segmentation targets are excluded by documented detection-only configuration.

## Full validation and future work

Do not start a complete trainval sensor download on the current 50 GB volume.
First enumerate validation payloads and sweeps from metadata, sum byte sizes,
include checkpoints/results and add 20% headroom. A bulk trainval strategy needs
a provisional terabyte-scale storage allowance plus extraction staging; this is
planning guidance, not a verified purchase size. A validation-only payload subset
may be smaller, but must retain all required sweeps. No resize/download performed.

After mini timing, estimate compute as:
`hourly_price * (N_val * (fusion_seconds + lidar_seconds) / 3600 + setup_hours)`.
Add data loading, export and evaluation time. No defensible measured per-sample
latency or full-val dollar forecast exists before running the stack.

Preserve prediction tokens, checkpoint/config/code hashes, payload/sweep and
LiDAR geometry identities for cache reuse. Reuse LiDAR clean predictions only
when all LiDAR inputs and processing are unchanged. Obtain P_F(0), P_L(0) first;
misalignment and direct LiDAR corruption remain distinct subsequent experiments.
