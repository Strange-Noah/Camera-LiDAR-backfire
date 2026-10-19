# BEVFusion mini inference: prepared, not GPU-validated

This environment is independent of `environment/metadata`. Do not install these
packages into Python 3.12. No training, finetuning, corruption, dataset migration,
full trainval download, or billable provisioning is performed by these files.

## Source decision

`upstream.json` pins official code to `db75150717a9462cb60241e36ba28d65f6908607`,
the parent of the BEVFusion-R integration (`d0152cf`). Current archived HEAD
`326653dc06e0938edf1aae7d01efcd158ba83de5` introduces Radar into the shared
detection config and converter. Using it unchanged would silently add a third
sensor data dependency. Current HEAD's download script supplies the refreshed
official URLs. Weight compatibility with the selected revision is **unverified**
until `smoke.py` completes a strict state-dict load and inference.

No upstream source patch is applied. Runtime config overrides are recorded:
batch size 1, zero loader workers for the first smoke test, mini infos path,
removal of unused `LoadBEVSegmentation` and `gt_masks_bev` collection, and disabling
backbone initialization download because the full detector checkpoint supplies it.
All image/LiDAR preprocessing stays intact: six views resized to 256x704, current
LiDAR frame plus nine sweeps, official calibration and pose transforms.

## Build candidate

From repository root on a Linux x86_64 Docker builder:

```bash
docker build -f environment/bevfusion/Dockerfile -t camera-lidar-bevfusion:db751507 .
```

This workstation has no Docker builder. The Dockerfile has **not** been built.
Alternatively start the official CUDA base as the Runpod container, bootstrap the
same OS prerequisites and Python 3.8 venv, then run `setup.sh` inside that container.
Do not assume nested Docker is available on a Pod. Container registry digest,
OS package manifest and complete pip freeze must be captured after a successful
build. Current runtime.txt pins are candidates, not an accepted lock.

Base: CUDA 11.3.1 Ubuntu 20.04, PyTorch 1.10.1+cu113, torchvision 0.11.2,
MMCV-full 1.4.0 (provides `mmcv`), MMDetection 2.20.0, Python 3.8.
The official Dockerfile's floating Miniconda and Python packages are replaced
with explicit primary pins. Installing both `mmcv` and `mmcv-full` is avoided.
Ubuntu supplies OpenMPI (version must be captured); official README lists 4.0.4.
The host driver must support CUDA 11.3 and the actual GPU. Runpod catalog CUDA
versions describe host compatibility, not which toolkit to install in this image.

Upstream extensions explicitly emit sm_70/75/80/86. On Ada, sm_86 cubins are
supported by NVIDIA; this does not prove every dependency or model kernel works.
Do not set CUDA_FORCE_PTX_JIT=1 as an acceptance requirement: upstream extensions
have compatible Ampere cubins without PTX. No Blackwell deployment is selected.

## Volume and permissions

Confirm attached volume is `p6vb29fmlu`, EU-RO-1, 50 GB, mounted at `/workspace`.
Run `du` before acquisition: `df` can show shared backend capacity rather than
the purchased quota. Maintain at least 10 GB quota headroom for mini preparation
and checkpoints; large build caches stay on the disposable container disk.
The original dataset remains root-owned and read-only. Re-run the existing
dataset protection verification and member setup on the new host. Require
Landlock ABI >=3; stop if unavailable. Do not relax permissions to make a job run.

Maintenance/root downloads the two official checkpoints only:

```bash
python environment/bevfusion/download_checkpoints.py
```

This writes `/workspace/checkpoints/bevfusion/{bevfusion-det.pth,lidar-only-det.pth}`
and a manifest containing source URL, byte length, SHA256, config and code commit.
The upstream does not publish a trusted hash in its download script: the recorded
hash is an acquired-artifact identity, not independently authenticated content.
Copy the small manifest to the feature branch after validation, never the weights.

Run preprocessing/inference as an established project member through the existing
protected experiment shell. Put the project checkout in a readable home location
for this member (`/root` is normally inaccessible); keep the root management copy
and `/opt/bevfusion` upstream separate. Example inside that shell, with `$USER`
being the established member login:

```bash
export PATH=/opt/venvs/bevfusion/bin:$PATH
export PYTHONPATH=/opt/bevfusion
python environment/bevfusion/prepare_mini.py --output /workspace/staging/$USER/bevfusion-mini
python environment/bevfusion/smoke.py --model fusion --limit 1 --data-view /workspace/staging/$USER/bevfusion-mini --output /workspace/results/$USER/fusion-one
python environment/bevfusion/smoke.py --model lidar --limit 1 --data-view /workspace/staging/$USER/bevfusion-mini --output /workspace/results/$USER/lidar-one
```

Use new output directories for each run, then `--limit 8` for smoke tests and
`--limit 0 --evaluate` for the entire mini_val split. Both models consume the same
timestamp-sorted infos; compare the saved tokens before comparing predictions.
Each model runs in a separate process. The custom runner initializes a one-rank
NCCL process group for upstream normalization layers. The official CLI alternative
is `torchpack dist-run -np 1 python tools/test.py CONFIG CHECKPOINT --out FILE`,
but requires equivalent data/config overrides; do not pass eight GPU ranks.

`prepare_mini.py` calls the official converter, bypassing the training-only GT DB
creation. Its symlink view contains the existing samples, sweeps, maps and mini
metadata, while pickle outputs live outside the immutable dataset. No raw files
are copied or modified. Mini metadata includes calibration, ego poses and labels.
All referenced payload files still need runtime verification on the attached host.

## Acceptance and cache contract

Capture `nvidia-smi`, `nvcc --version`, `pip freeze`, build log and imported ops.
The runner strictly loads checkpoint tensors, executes the actual CUDA paths,
and saves raw predictions, nuScenes global-coordinate export, resolved config,
sample tokens, checkpoint/infos hashes, timings, GPU allocator peaks and process
peak RSS. Use `nvidia-smi` sampling for total device memory in addition to allocator
statistics. Process RSS with workers=0 covers the data-loading process; it is not
host-wide RAM. Timings include transfer/model/postprocessing, exclude loader wait;
wall time also includes setup. Inspect first-sample box sizes, axes, classes and
scores against the sensor visualization before declaring Stage 3 passed.

Any exception writes a failed report and exits nonzero. A successful load is not
the same as a successful smoke test. Mini metrics are never reported as official
full-val replication. Do not evaluate a partial split with the full split evaluator.

Future cached LiDAR predictions may be reused only when sample tokens, LiDAR
payloads/sweeps, LiDAR poses/transforms, preprocessing, weights and code/config
identities are unchanged. Camera/extrinsic-only perturbation and LiDAR corruption
must have different experiment identities. No corruption code is included here.
