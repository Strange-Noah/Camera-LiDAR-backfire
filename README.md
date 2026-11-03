# Camera–LiDAR Backfire

Research on camera–LiDAR fusion robustness under sensor misalignment, with a focus on identifying conditions under which sensor fusion may perform worse than an individual sensor.

**Current status: CONDITIONAL GO.** The official 850-scene nuScenes trainval metadata-only audit is complete. Separately, a BEVFusion MMCV 2.1/CUDA 12.8 Docker inference workflow and an exploratory temporal camera-misalignment study have been run. The temporal study has a candidate, operating-point-sensitive backfire result; it has not established a general threshold. MultiCorrupt integration, matched-precision evaluation, and independent confirmation remain future work.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What Is Complete](#what-is-complete)
- [Windows Conda Environment for Metadata Audit](#windows-conda-environment-for-metadata-audit)
- [MMCV 2.1 / CUDA 12.8 Migration](#mmcv-21-cuda-128-migration-details)
- [Docker, Dataset, and Temporal Experiment](#completed-mmcv-21-and-temporal-experiment-record)
- [Full Metadata Audit Results](#full-metadata-audit-results)
- [Temporal Inference Results](#aggregate-metrics-obtained)
- [Object-Level Analysis and Backfire Findings](#object-level-findings-and-what-they-establish)
- [Limitations and Next Steps](#what-this-does-and-does-not-show)
- [Data and Git Policy](#data-and-git-policy)

---

## Project Overview

The project investigates the robustness of camera–LiDAR sensor fusion under spatial and temporal sensor misalignment.

The completed metadata audit provides an initial assessment of whether the dataset and its annotations support the proposed investigation. It does not establish that camera–LiDAR fusion outperforms either sensor individually or demonstrate that fusion backfires under any particular perturbation.

The completed metadata audit and the model experiment are separate evidence tracks. The BEVFusion checkout has been migrated sufficiently to run pretrained inference, and the temporal experiment has compared clean and shifted camera inputs with a matched LiDAR-only baseline on a common validation subset. See the detailed [temporal experiment record](#completed-mmcv-21-and-temporal-experiment-record) and [MMCV 2.1/CUDA 12.8 migration details](#mmcv-21-cuda-128-migration-details).

Remaining model-research work includes fair matched-precision comparison, confirmatory analysis on independent labelled data if available, and MultiCorrupt integration.

## Windows Conda Environment for Metadata Audit

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


## Windows Dataset Acquisition


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
python scripts/verify_nuscenes.py --render-once
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

## BEVFusion Temporal Misalignment Status

A separate model experiment was run in Docker with PyTorch 2.7.1+cu128, MMCV 2.1.0, MMEngine 0.10.7, and MMDetection 3.3.0. It generated temporal manifests and predictions for seven camera timing conditions plus LiDAR-only, using 5,063 common nuScenes validation samples.

Aggregate fusion mAP/NDS remained above LiDAR-only at every tested offset. Object-level analysis identified fast cars 40–60 m away as a candidate group where large shifts can reduce recall below LiDAR-only at a fixed score cutoff of 0.2. This comparison is cutoff-sensitive and exploratory; no general backfire threshold has been established. These model results are not part of the 850-scene metadata audit and must not be described as official full-validation benchmark results.

See the [temporal experiment record](#completed-temporal-experiment-record).

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

- The full nuScenes dataset or trainval sensor blobs.
- Model weights and checkpoints.
- Credentials, tokens, or other secrets.

Experiment results, manifests, prediction files, evaluation reports, and derived analysis tables under `results/` are intended to be versioned and pushed with the project. Check available GitHub storage limits before adding exceptionally large result artifacts.


```text
results/metadata_audit/
```

The generated experiment outputs in `results/` are versioned so teammates can inspect and reproduce the reported findings.

For future experiments, keep the raw dataset as canonical input and store generated outputs under the appropriate results or checkpoint directory. Preserve the existing dataset rather than using it as a destination for experiment outputs.

---

## Documentation and Reproduction

This is the single canonical project guide. Supporting non-README files below contain machine-readable audit details and methods.

| Document | Purpose |
|---|---|
| [Machine-readable gates](results/metadata_audit/gating_report.json) | Machine-readable research gates |
| [Methods, validity rules, and reproduction commands](docs/metadata_audit_methods.md) | Audit methodology and reproduction instructions |

The machine-readable gates and methods remain available in results/metadata_audit/gating_report.json and docs/metadata_audit_methods.md.

**Current research milestone:** The metadata audit is complete. BEVFusion migration, pretrained inference, and exploratory temporal misalignment comparisons have been run in the nested checkout. Matched-precision and independent confirmatory analyses, plus MultiCorrupt integration, remain future work.

---

## MMCV 2.1 CUDA 12.8 Migration Details

This guide describes the Windows + Docker Desktop workflow used for this BEVFusion checkout and an RTX 50-series GPU. The upstream [README](README.md) documents the original MMCV 1.x / MMDetection 2.x stack; these instructions apply to the migrated setup.

## Status

- Reported environment: PyTorch `2.7.1+cu128`, MMCV `2.1.0`, MMEngine `0.10.7`, MMDetection `3.3.0`.
- Package and converter imports succeeded; NuScenes info files were generated.
- A separate standard Camera + LiDAR run processed all 6,019 validation samples and formatted predictions; Docker Desktop returned an engine API error before that run reported final metrics. The temporal experiment later completed saved-prediction evaluation on its matched 5,063-sample subset, with results reported below.
- Training under the migrated environment has not been validated.

## 1. Check GPU access and build

In PowerShell:

```powershell
docker context use desktop-linux
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

Set-Location .\bevfusion
docker build -t bevfusion:mmcv21 -f .\docker\Dockerfile .
```

The Dockerfile uses CUDA 12.8, Python 3.10, PyTorch 2.7.1, MMCV 2.1.0, and MMDetection 3.3.0. It pins `setuptools<81` because MMCV source builds need `pkg_resources`.

## 2. Start an interactive container

From the repository root, enter the BEVFusion folder and provide the host folder containing samples, sweeps, maps, and v1.0-trainval:

```powershell
Set-Location .\bevfusion
$projectPath = (Resolve-Path .).Path
$nuscenesPath = Read-Host "Enter the full path to your nuScenes folder"
$nuscenesPath = (Resolve-Path $nuscenesPath).Path

docker run --rm -it --gpus all --shm-size=8g `
  -v "${projectPath}:/workspace/bevfusion" `
  -v "${nuscenesPath}:/workspace/nuscenes" `
  -w /workspace/bevfusion `
  --entrypoint bash `
  bevfusion:mmcv21
```

Mounting the dataset keeps it available when you create another container. Inside the container, connect it to the config's default path if `data/nuscenes` does not already exist:

```bash
mkdir -p data
ln -s /workspace/nuscenes data/nuscenes
ls data/nuscenes/v1.0-trainval
```

For an existing stopped container, try `docker start -ai <name-or-id>`. To open another shell in a running container, use `docker exec -it <name-or-id> bash`. If a stopped container exits immediately, start a new interactive shell with the `docker run` command above.

## 3. Verify dependencies and build CUDA extensions

Run inside `/workspace/bevfusion`:

```bash
python3 -c "import torch, mmcv, mmengine, mmdet; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.is_available()); print('mmcv', mmcv.__version__, 'mmengine', mmengine.__version__, 'mmdet', mmdet.__version__)"
python3 -c "import sys; sys.path.insert(0, 'tools'); from data_converter import nuscenes_converter; from data_converter.create_gt_database import create_groundtruth_database; print('converter imports OK')"
```

Build the project's custom operators in this same container if their shared libraries are missing:

```bash
python3 setup.py develop
find mmdet3d/ops -name '*.so' | wc -l
```

`setup.py` defines 12 custom extension modules. Rebuild them after changing PyTorch, CUDA, or the GPU architecture environment.

## 4. Prepare NuScenes (skip if already prepared)

Validation requires the raw NuScenes data and `data/nuscenes/nuscenes_infos_val.pkl`. Training also uses the generated ground-truth database. The full prep command creates info files and attempts to build that database; it can take hours and use substantial memory:

```bash
python3 tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes
```

Pretrained inference does not require the ground-truth database. If the validation info file and raw validation data already exist, do not rerun full prep just to infer.

## 5. Run pretrained inference and NuScenes evaluation

Run directly with Python; this migrated single-GPU test runner does not need `torchpack dist-run`. Set data workers to zero to reduce memory pressure (loading may be slower). `jsonfile_prefix` keeps formatted JSON and metrics in the mounted project folder instead of temporary `/tmp` storage.

### LiDAR-only TransFusion-L

```bash
python3 tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
  pretrained/lidar-only-det.pth \
  --eval bbox --out lidar_results.pkl \
  --cfg-options data.workers_per_gpu=0 \
  --eval-options jsonfile_prefix=lidar_results_formatted/results
```

### Camera + LiDAR BEVFusion

```bash
python3 tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth \
  --eval bbox --out bevfusion_results.pkl \
  --cfg-options data.workers_per_gpu=0 \
  --eval-options jsonfile_prefix=bevfusion_results_formatted/results
```

The prediction pickle and formatted NuScenes JSON are saved in the project folder. Keep the terminal open until metric output appears. In the reported run, Camera + LiDAR inference took about 2 hours 45 minutes for 6,019 samples; runtime depends on GPU and data-loading speed.

## 6. Evaluate a saved prediction pickle without rerunning inference

If inference saved its pickle but evaluation was interrupted, run this from the project root inside the container. It formats the saved predictions to a persistent JSON path, then evaluates that file:

```bash
python3 - <<'PY'
import pickle
from mmengine.config import Config
from mmengine.registry import DefaultScope
from torchpack.utils.config import configs
from mmdet3d.datasets import build_dataset
from mmdet3d.utils import recursive_eval

config_path = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
configs.load(config_path, recursive=True)
cfg = Config(recursive_eval(configs), filename=config_path)
DefaultScope.get_instance("bevfusion-eval-only", scope_name="mmdet")
dataset = build_dataset(cfg.data.test)
with open("bevfusion_results.pkl", "rb") as f:
    predictions = pickle.load(f)
result_files, _ = dataset.format_results(
    predictions, jsonfile_prefix="bevfusion_results_formatted/results"
)
result_path = result_files["pts_bbox"] if isinstance(result_files, dict) else result_files
print(dataset._evaluate_single(result_path))
PY
```

For LiDAR-only, change the config, pickle, and output names to those shown above. Avoid opening the full JSON in a text editor while evaluation runs; parsing it can need substantial memory.

## Troubleshooting

- **`No module named mmdet3d`**: run from the repository root and use `python3 setup.py develop` if the editable project install or custom operators are missing.
- **Legacy `mmdet.core`, `mmcv.Config`, or `mmcv.dump` imports**: don't create dummy modules inside site-packages. Use the compatibility changes in this repository and MMEngine equivalents.
- **Voxelization crash or missing extension**: check `torch.cuda.is_available()` and rebuild the custom CUDA extensions in the active container.
- **Camera depth dtype/channel/signature errors**: keep the migrated fixes in `mmdet3d/models/vtransforms/base.py` and `depth_lss.py`.
- **Docker `500 Internal Server Error`, WSL error, or `unexpected EOF`**: check Docker Desktop/WSL and `docker ps -a`. If the prediction pickle exists, use the saved-prediction evaluation section before rerunning inference.
- **Memory pressure**: use `data.workers_per_gpu=0`; the GT database is unnecessary for pretrained inference.


---

## Completed MMCV 2.1 and temporal experiment record

This section records the completed commands and findings from the Docker run. It supplements the general pilot recipe above; where example settings conflict, the exact completed pilot/full-val settings below are authoritative.

### Environment and data preparation

The Linux Docker context, CUDA GPU access, image build, and interactive container were started from PowerShell:

    docker context use desktop-linux
    docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
    cd .\bevfusion
    docker build -t bevfusion:mmcv21 -f .\docker\Dockerfile .
    docker run --rm -it --gpus all --shm-size=8g -v "$($PWD.Path):/workspace/bevfusion" -w /workspace/bevfusion bevfusion:mmcv21 bash

Inside the container, reported versions were PyTorch 2.7.1+cu128, CUDA 12.8, MMCV 2.1.0, MMEngine 0.10.7, and MMDetection 3.3.0. Converter imports succeeded. The custom project operators were compiled with python3 setup.py develop; 12 .so files were found. A standalone CUDA voxelization check on the RTX 5060 succeeded. This does not validate the training path.

The data-preparation command used was:

    python3 tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes

It reported 850 scenes (700 train, 150 val), 34,149 samples, and wrote train/val info pickles of about 1.2 GB and 259 MB. The GT database step was killed by memory pressure on one attempt and later hit a malformed sweep file. The info pickles existed and were sufficient for pretrained inference; pretrained inference does not require the GT database. The mounted dataset was linked with:

    ln -sfn /workspace/nuscenes data/nuscenes

For package installation, CUDA extensions, and migration fixes, see the [MMCV 2.1 CUDA 12.8 migration details](#mmcv-21-cuda-128-migration-details). Do not create fake mmdet.core stubs in site-packages.

### Timestamp audit and manifests actually used

The timestamp audit command was:

    python3 experiments/temporal_misalignment/audit_timestamps.py --dataroot /workspace/nuscenes

Median camera spacing was 100 ms (some intervals were 50 ms). Median clean camera–LiDAR deltas were: CAM_FRONT −35.9 ms; CAM_FRONT_RIGHT −27.9 ms; CAM_FRONT_LEFT −43.5 ms; CAM_BACK −10.7 ms; CAM_BACK_LEFT −0.9 ms; CAM_BACK_RIGHT −20.3 ms.

The pilot command actually used was:

    python3 experiments/temporal_misalignment/prepare_manifests.py \
      --dataroot /workspace/nuscenes \
      --info /workspace/nuscenes/nuscenes_infos_val.pkl \
      --load-interval 60 \
      --offsets-ms -100 0 100 \
      --tolerance-ms 50 \
      --out-dir results/temporal_pilot/manifests

It began with 101 samples and retained 91 common tokens.

The full-val command actually used was:

    python3 experiments/temporal_misalignment/prepare_manifests.py \
      --dataroot /workspace/nuscenes \
      --info /workspace/nuscenes/nuscenes_infos_val.pkl \
      --load-interval 1 \
      --offsets-ms -300 -200 -100 0 100 200 300 \
      --tolerance-ms 50 \
      --out-dir results/temporal_fullval/manifests

It retained 5,063 common samples from 6,019 validation samples. The token list is results/temporal_fullval/manifests/common_sample_tokens.json. Each condition has 30,378 manifest rows (5,063 samples × 6 cameras).

Actual shifts are quantized and camera-dependent. Requested −200 ms had median realized shift −250 ms (range −250 to −150 ms). Requested −300 ms had camera medians around −300 or −350 ms (range −350 to −250 ms). Requested +200 ms had camera medians around +150 to +200 ms. Requested +300 ms had median +250 ms for all cameras (range +250 to +350 ms). Report realized offsets from the manifest, not only the requested shift.

### Temporal inference and saved evaluation

Fusion config/checkpoint:

    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
    pretrained/bevfusion-det.pth

LiDAR-only config/checkpoint:

    configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
    pretrained/lidar-only-det.pth

The temporal runner command format (shown for +100 ms) was:

    python3 experiments/temporal_misalignment/run_test.py \
      --manifest results/temporal_fullval/manifests/manifest_plus100ms.csv \
      --dataroot /workspace/nuscenes \
      --common-tokens results/temporal_fullval/manifests/common_sample_tokens.json \
      -- \
      configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
      pretrained/bevfusion-det.pth \
      --out results/temporal_fullval/plus100/predictions.pkl

Use the corresponding manifest and condition folder for minus300, minus200, minus100, zeroms, plus100, plus200, and plus300. For a matched LiDAR-only baseline, the same common-token list and zeroms manifest were used with the LiDAR config/checkpoint:

    python3 experiments/temporal_misalignment/run_test.py \
      --manifest results/temporal_fullval/manifests/manifest_zeroms.csv \
      --dataroot /workspace/nuscenes \
      --common-tokens results/temporal_fullval/manifests/common_sample_tokens.json \
      -- \
      configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
      pretrained/lidar-only-det.pth \
      --out results/temporal_fullval/lidar_only/predictions.pkl

All eight conditions produced 5,063 predictions. To evaluate a saved prediction pickle without repeating inference:

    python3 experiments/temporal_misalignment/evaluate_saved.py \
      --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
      --predictions results/temporal_fullval/plus100/predictions.pkl \
      --common-tokens results/temporal_fullval/manifests/common_sample_tokens.json \
      --jsonfile-prefix results/temporal_fullval/plus100/evaluation/results_nusc

Change the pickle and JSON prefix for each condition and use the LiDAR config for lidar_only. A full evaluation once ran out of memory while loading nuScenes split tables; evaluating saved predictions in a fresh process succeeded. Avoid opening the large formatted JSON in a text editor during evaluation.

### Aggregate metrics obtained

Pilot values are from 91 common samples and are only pipeline/subset diagnostics:

| Condition | mAP | NDS |
|---|---:|---:|
| LiDAR-only | 0.6813 | 0.7205 |
| Clean fusion | 0.7086 | 0.7334 |
| −100 ms | 0.7148 | 0.7360 |
| +100 ms | 0.7113 | 0.7336 |

Full temporal sweep values are from the 5,063 common-token subset, not all 6,019 validation samples:

| Condition | mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| −300 ms | 0.6675 | 0.2909 | 0.2566 | 0.3169 | 0.2220 | 0.1808 | 0.7070 |
| −200 ms | 0.6693 | 0.2899 | 0.2559 | 0.3165 | 0.2249 | 0.1809 | 0.7078 |
| −100 ms | 0.6712 | 0.2885 | 0.2558 | 0.3133 | 0.2289 | 0.1799 | 0.7090 |
| 0 ms clean | 0.6725 | 0.2873 | 0.2558 | 0.3093 | 0.2324 | 0.1792 | 0.7098 |
| +100 ms | 0.6720 | 0.2885 | 0.2562 | 0.3112 | 0.2278 | 0.1800 | 0.7096 |
| +200 ms | 0.6703 | 0.2891 | 0.2565 | 0.3121 | 0.2253 | 0.1803 | 0.7088 |
| +300 ms | 0.6699 | 0.2907 | 0.2567 | 0.3128 | 0.2246 | 0.1812 | 0.7083 |
| LiDAR-only | 0.6481 | 0.2922 | 0.2532 | 0.3149 | 0.2461 | 0.1805 | 0.6954 |

Aggregate fusion mAP/NDS exceeded LiDAR-only at every offset. Thus aggregate metrics show degradation from clean timing at larger shifts, but not aggregate backfire.

### Object-level analysis commands and method

The full object analysis used normalized detection class names, GT boxes with LiDAR points, finite GT velocity for speed bins, and score-ordered same-class XY-center matching. Initial diagnostics used score thresholds 0.1/0.3/0.5 and center radii 0.5/1/2/4 m. The later sensitivity sweep used six cutoffs and a 2 m radius:

    python3 experiments/temporal_misalignment/analyze_object_level.py \
      --info /workspace/nuscenes/nuscenes_infos_val.pkl \
      --common-tokens results/temporal_fullval/manifests/common_sample_tokens.json \
      --manifest results/temporal_fullval/manifests/manifest_zeroms.csv \
      --results-dir results/temporal_fullval \
      --score-thresholds 0.15 0.20 0.25 0.30 0.35 0.40 \
      --center-thresholds 2.0 \
      --out-dir results/temporal_fullval/object_analysis/score_sweep

These fixed-score subgroup recall results are not nuScenes AP. Scene-level bootstrap comparisons over class/speed/distance bins were run as:

    python3 experiments/temporal_misalignment/bootstrap_object_groups.py \
      --matches results/temporal_fullval/object_analysis/score_sweep/per_gt_matches.csv \
      --out results/temporal_fullval/object_analysis/all_groups_vs_lidar.csv \
      --classes car truck bus trailer construction_vehicle pedestrian motorcycle bicycle \
      --speed-bins 0-1mps 1-5mps 5mps+ \
      --distance-bins 0-20m 20-40m 40-60m 60m+ \
      --score-thresholds 0.2 0.3 0.4 \
      --center-threshold 2.0 \
      --reference-condition lidar_only \
      --bootstrap 5000 \
      --seed 42

The focused fast-car six-cutoff tables used the same command with class car, speed-bin 5mps+, distance-bin 40-60m, cutoffs 0.15/0.20/0.25/0.30/0.35/0.40, and reference-condition set once to lidar_only and once to zeroms. When reference-condition is zeroms, the script's column named lidar_recall actually contains clean-fusion recall; interpret it using the reference_condition column.

### Object-level findings and what they establish

The strongest current candidate is fast cars (>5 m/s) at 40–60 m: 1,750 objects in 127 scenes. With a 2 m XY-center radius and score threshold 0.2:

| Requested shift | Fusion recall | LiDAR-only recall | Difference | Scene-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| −300 ms | 0.6000 | 0.6246 | −2.46 pp | [−4.67, −0.30] pp |
| −200 ms | 0.6240 | 0.6246 | −0.06 pp | [−2.25, +2.19] pp |
| +200 ms | 0.6234 | 0.6246 | −0.11 pp | [−1.95, +1.82] pp |
| +300 ms | 0.6023 | 0.6246 | −2.23 pp | [−4.06, −0.33] pp |

Against clean fusion at cutoff 0.2, shifted-minus-clean recall was −3.60 pp at −300 ms (CI [−5.12, −2.10]), −1.20 pp at −200 ms (CI [−2.32, −0.07]), −1.26 pp at +200 ms (CI [−2.34, −0.17]), and −3.37 pp at +300 ms (CI [−5.03, −1.87]).

The score-cutoff sweep changes the backfire interpretation: at cutoff 0.2, the tested −300 and +300 ms conditions were below LiDAR-only with intervals below zero. At cutoffs 0.15, 0.25, 0.30, 0.35, and 0.40, those comparisons were inconclusive or favored fusion. Shifted fusion was lower than clean fusion across the reported fast-car rows. Therefore there is evidence of temporal-shift degradation relative to clean fusion, while fusion-below-LiDAR backfire is operating-point-sensitive.

At cutoff 0.2, the candidate crossing is bracketed between requested ±200 and ±300 ms, but not precisely located. The realized medians are around −250 ms for requested −200, around −300/−350 ms for requested −300, around +150/+200 ms for requested +200, and around +250 ms for requested +300. Analyze early and late shifts separately.

A weaker exploratory candidate was bicycles at 1–5 m/s and 0–20 m: 56 objects in 8 scenes. At score 0.3 and 2 m radius, +200 ms recall was 0.7321 versus 0.7857 for both clean fusion and LiDAR-only. The estimate was sensitive to removing the largest scene. Three inspected cases also showed score changes with predicted centers remaining near GT; these are illustrations, not causal proof.

### Current conclusion and next work

The project has established a working Docker MMCV 2.1/CUDA 12.8 inference workflow, timestamp-aware manifests, matched predictions for seven fusion conditions and LiDAR-only, aggregate evaluation, and exploratory object-level comparisons.

Current scoped result: **fast, distant cars are a candidate conditional backfire group at fixed score cutoff 0.2 under the tested large shifts.** The overall matched-subset mAP/NDS remains higher for fusion than LiDAR-only, and the subgroup backfire does not persist at other tested cutoffs. No general or cutoff-independent temporal threshold is established. Many subgroups, offsets, and cutoffs were searched, so this is exploratory and the displayed bootstrap intervals are not adjusted for that search.

Next: choose a fair fusion-versus-LiDAR operating point, preferably recall at matched overall precision; selecting model-specific cutoffs requires counting false positives, which the per-GT file alone does not provide. Choose cutoffs on calibration scenes and compare subgroup recall on separate scenes. Keep shifted-vs-clean (temporal degradation) separate from shifted-vs-LiDAR (backfire). Only if the fast-car effect persists should you refine the shift bracket with intermediate manifests, checking actual offsets and distinct selected camera tokens before running expensive inference. Preserve and commit the manifests, predictions, evaluation files, and derived analysis outputs; keep the full dataset, checkpoints, and secrets outside Git.
