# Camera–LiDAR Backfire

Research on camera–LiDAR fusion robustness under sensor misalignment, with a focus on identifying conditions under which sensor fusion may perform worse than an individual sensor.

**Current status: CONDITIONAL GO for the metadata audit; calibration remains exploratory.** The official 850-scene nuScenes trainval metadata-only audit is complete. Separately, a BEVFusion MMCV 2.1/CUDA 12.8 Docker inference workflow and an exploratory temporal camera-misalignment study have been run. The temporal study has a candidate, operating-point-sensitive backfire result; it has not established a general threshold. MultiCorrupt integration, matched-precision evaluation, and independent confirmation remain future work.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What Is Complete](#what-is-complete)
- [Windows Conda Environment for Metadata Audit](#windows-conda-environment-for-metadata-audit)
- [MMCV 2.1 / CUDA 12.8 Migration](#mmcv-21-cuda-128-migration-details)
- [Docker, Dataset, and Temporal Experiment](#completed-mmcv-21-and-temporal-experiment-record)
- [Full Metadata Audit Results](#full-metadata-audit-results)
- [Temporal Inference Results](#aggregate-metrics-obtained)
- [OCaMo Calibration Experiments](#cameralidar-calibration-experiments)
  - [Frozen multi-camera consensus gate evaluation](#frozen-multi-camera-consensus-gate-evaluation)
  - [Frozen score-curve gate evaluation](#frozen-score-curve-gate-evaluation)
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

Current scoped result: **fast, distant cars are a candidate conditional backfire group at fixed score cutoff 0.2 under the tested large shifts.** Overall matched-subset mAP/NDS remains higher for fusion than LiDAR-only, and the subgroup backfire does not persist at other tested cutoffs. No general or cutoff-independent temporal threshold is established. Many subgroups, offsets, and cutoffs were searched, so the displayed bootstrap intervals are exploratory and are not adjusted for that search.

Next: choose a fair fusion-versus-LiDAR operating point, preferably recall at matched overall precision; selecting model-specific cutoffs requires counting false positives, which the per-GT file alone does not provide. Choose cutoffs on calibration scenes and compare subgroup recall on separate scenes. Keep shifted-versus-clean temporal degradation separate from shifted-versus-LiDAR backfire. Only if the fast-car effect persists should the shift bracket be refined with intermediate manifests, checking actual offsets and distinct selected camera tokens before running expensive inference. Preserve manifests, predictions, evaluation files, and derived analysis outputs; keep the full dataset, checkpoints, and secrets outside Git.

## Camera–LiDAR Calibration Experiments

This section records a separate OCaMo/LTO-inspired camera–LiDAR rotational-calibration study. It asks whether projected LiDAR edge structure can indicate and recover a known synthetic camera–LiDAR roll error. It does not measure temporal-offset recovery or object-level detection backfire.

### OCaMo/LTO and nuScenes checks

- Ran the released OCaMo/LTO monitoring and drift-tracking examples on their supplied CARLA snippet. They generated `img/ocamo_monitoring.png`, `img/ocamo_drift_tracking.png`, `img/lto_monitoring.png`, and `img/lto_drift_tracking.png`; `img/example_projection.png` shows the example projection. This verifies the example scripts execute, not nuScenes calibration.
- Audited nuScenes metadata: 850 scenes, 34,149 samples, and 204,894 camera/LiDAR records.
- A projection diagnostic processed 34,720 LiDAR points, found 32 ring IDs, extracted 9,232 OCaMo corners, and retained 711 inside the image. Median projected-corner distance to an image edge was 15.06 px; 21.1% were within 3 px and 29.0% within 5 px. These are geometry diagnostics, not calibration accuracy.
- The inspected LiDAR scan had ring IDs 0–31. Adjacent azimuth steps had median 0.000734 rad and p95 1.401 rad; 12.0% exceeded 0.1 rad. Ring-wise corner ordering is approximate and scan discontinuities may affect the edge objective.
- Fixed rotation-score sweeps tested roll, pitch, and yaw candidate perturbations. The score measures projected-corner/image-edge agreement; it is not detector performance or proof that the score peak is an unbiased calibration estimate.

#### Reproduce the released examples and geometry checks

The four released scripts use OCaMo's supplied CARLA example data. From the mounted OCaMo checkout:

```bash
cd /workspace/OCaMo/python/inference
python3 -c "import numpy, scipy, cv2, matplotlib"
python3 ocamo_monitoring.py
python3 ocamo_drift_tracking.py
python3 lto_monitoring.py
python3 lto_drift_tracking.py
```

For the nuScenes projection diagnostics from the BEVFusion checkout:

```bash
cd /workspace/bevfusion
python3 experiments/calibration_baseline/check_nuscenes_projection.py
python3 experiments/calibration_baseline/check_projection_sequence.py
python3 experiments/calibration_baseline/sweep_rotation_score.py
```

The image outputs and sweep CSVs are written below `results/calibration_baseline/`. The source currently selects a configured scene for the fixed-score sweep; consult the script if changing the scene.

### Online roll tracker and experimental controls

`experiments/calibration_baseline/run_online_roll_tracker.py` injects a synthetic rotation into the camera-from-LiDAR transform, extracts ring-ordered LiDAR corners, and estimates roll by comparing projected corners with an image-edge distance field. The tested injections are 0°, +0.5°, and −0.5°; the expected corrections are respectively 0°, −0.5°, and +0.5°. The inverse-transform checks were at floating-point precision (about 0–2.84e−14 m), validating transform composition arithmetic, not calibration accuracy.

The tracker uses a short sequence of 40 frames for scenes 0001–0004 and 39 for scene 0005. Paired summaries use frames with `frame >= 10`, for 149 frames per condition across five scenes. Finite point/corner filtering was added; ring IDs are range-checked before conversion. GT moving-object filtering, when enabled, uses annotations and is an oracle. `USE_GT_MOVING_FILTER=0` disables that filter and skips loading the nuScenes info file for it.

Support modes compared:

| Mode | How support points are selected | GT/clean information used for support? |
|---|---|---|
| `clean` | Visible under the recorded clean transform | Clean transform is an oracle; GT filtering depends on the run setting. |
| `runtime` | Visible under the perturbed starting transform | No clean transform for support selection; GT filtering depends on the run setting. |
| `runtime_matched` | Seeded random subset of runtime support, downsampled to the search-stable count | No clean transform for support selection; count-matched control, one seed in the original comparison. |
| `search_stable` | Runtime-visible points that remain visible across the correction search range (about ±0.951°, sampled at 0.05°) | No clean transform or GT labels for support selection. |

The GT-filter ablation showed that enabling/disabling the moving-object filter had only a small, mixed effect; it did not account for the large difference between clean and runtime support. The runtime-matched control retained the same per-scene counts as search-stable support but recovered less accurately, suggesting support composition matters more than count alone. Runtime-matched was initially tested with one random seed; later seeded runs are described in the README’s experiment records where applicable.

### Earlier four-way support and GT-filter ablation

The first online-tracker ablation used five scenes, both support sources (`clean` and `runtime`), GT moving-object filtering on/off, and both injection signs. These are descriptive frame-paired results over 149 post-burn-in frames per condition. `Target beats zero` compares the target candidate correction score with a zero correction; `target grid winner` is the fraction of frames where the known target was the highest-scoring candidate on the tested grid.

| Support | GT filter | Injection | Mean paired change (target) | MAE | Correct direction | Target beats zero | Target grid winner | Support mismatches vs clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | On | +0.5° | −0.300° (−0.5°) | 0.201° | 100.0% | 82.6% | 21.5% | 0 / 149 |
| Clean | On | −0.5° | +0.374° (+0.5°) | 0.148° | 97.3% | 87.2% | 17.4% | 0 / 149 |
| Clean | Off | +0.5° | −0.291° (−0.5°) | 0.211° | 100.0% | 83.2% | 21.5% | 0 / 149 |
| Clean | Off | −0.5° | +0.376° (+0.5°) | 0.153° | 96.0% | 87.2% | 18.1% | 0 / 149 |
| Runtime | On | +0.5° | −0.190° (−0.5°) | 0.310° | 90.6% | 37.6% | 7.4% | 144 / 149 |
| Runtime | On | −0.5° | +0.057° (+0.5°) | 0.443° | 59.1% | 38.9% | 3.4% | 140 / 149 |
| Runtime | Off | +0.5° | −0.178° (−0.5°) | 0.322° | 92.6% | 38.3% | 7.4% | 142 / 149 |
| Runtime | Off | −0.5° | +0.074° (+0.5°) | 0.426° | 72.5% | 38.3% | 4.0% | 140 / 149 |

This ablation indicated that clean-transform support substantially outperformed runtime support. Turning GT moving-object filtering on/off had only a small, mixed effect compared with the support-selection difference. Clean-transform support is oracle-assisted and is not a deployable support strategy.
### Five-scene support results

For the kernel-sum objective, GT filter off, and paired post-burn-in frames, the equal-scene means were:

| Support | Injection (target correction) | Mean paired change | MAE to target | Correct direction | Mean support corners |
|---|---|---:|---:|---:|---:|
| Runtime | +0.5° (−0.5°) | −0.203° | 0.297° | 93.3% | 163.4 |
| Runtime-matched | +0.5° (−0.5°) | −0.215° | 0.285° | 97.3% | 152.1 |
| Search-stable | +0.5° (−0.5°) | −0.360° | 0.142° | 100.0% | 152.1 |
| Runtime | −0.5° (+0.5°) | +0.083° | 0.433° | 57.8% | 173.3 |
| Runtime-matched | −0.5° (+0.5°) | +0.072° | 0.444° | 56.5% | 161.9 |
| Search-stable | −0.5° (+0.5°) | +0.349° | 0.165° | 90.7% | 161.9 |


For comparison, the robust-Chamfer objective with GT filtering off gave these five-scene equal-scene means:

| Support | Injection (target correction) | Mean paired change | MAE to target | Correct direction | Mean support corners |
|---|---|---:|---:|---:|---:|
| Runtime | +0.5° (−0.5°) | −0.178° | 0.322° | 92.6% | 163.1 |
| Search-stable | +0.5° (−0.5°) | −0.311° | 0.198° | 100.0% | 151.8 |
| Runtime | −0.5° (+0.5°) | +0.074° | 0.426° | 72.5% | 172.9 |
| Search-stable | −0.5° (+0.5°) | +0.257° | 0.251° | 81.9% | 161.6 |

The improvement pattern therefore appeared under both robust-Chamfer and kernel-sum objectives, while effect sizes differed. The clean-state issue remains material.
Search-stable support improved the synthetic injection response relative to runtime and runtime-matched support, especially for the negative injection. However, clean search-stable runs showed a positive bias: mean tracker estimate +0.205°, mean absolute estimate from zero 0.205°, and only 18.8% of score peaks within ±0.1° of zero. Runtime support had +0.035° mean estimate, 0.062° mean absolute estimate, and 40.3% of score peaks within ±0.1°. This clean-state behavior prevents a broad reliability claim from the five-scene test.

#### Reproduce the five-scene kernel-sum support comparison

Inside the prepared Docker container, from `/workspace/bevfusion`, the runtime and search-stable conditions were run for each scene and injection as follows:

```bash
cd /workspace/bevfusion

for i in 0 1 2 3 4; do
  for support in runtime search_stable; do
    for injection in 0 0.5 -0.5; do
      SCENE_INDEX=$i \
      SUPPORT_MODE=$support \
      USE_GT_MOVING_FILTER=0 \
      OBJECTIVE_MODE=kernel_sum \
      INJECTED_ROLL_DEG=$injection \
        python3 experiments/calibration_baseline/run_online_roll_tracker.py
    done
  done
done
```

The runtime-matched control used `SUPPORT_MODE=runtime_matched` with the same scene/injection loop. The main result CSVs are stored under `results/calibration_baseline/`; their filenames record objective, injection, support mode, filter setting, and scene.

### Multi-camera selective correction gate

The frozen six-camera gate and its additional scene evaluation are documented in [Frozen multi-camera consensus gate evaluation](#frozen-multi-camera-consensus-gate-evaluation). In brief, it smooths per-camera estimates, requires four-camera sign agreement, a minimum median magnitude, and three stable consecutive frames. It rejected every clean frame in the reported batches and had 100% correct direction among accepted synthetic-injection cases, but accepted only a minority of injected frames. The result supports selective correction under the tested synthetic ±0.5° roll settings, not a generally reliable calibration method.

Earlier tracker commands, detailed ablations, and per-scene outputs are retained in the experiment record and generated CSVs under `results/calibration_baseline/`. The current calibration status and next steps are summarized below the latest gate results.

### Frozen multi-camera consensus gate evaluation

This is the latest calibration experiment. It applies a frozen, runtime-only consensus gate to the six per-camera roll estimates from the kernel-sum tracker. It is separate from the earlier single-camera support-selection ablations. The calibration harness injects synthetic roll for evaluation; support selection uses the perturbed starting transform and no ground-truth moving-object filter.

The gate uses these fixed settings:

- Median smooth each camera estimate over five frames.
- Require at least four of six cameras to agree on the correction sign.
- Require the six-camera median magnitude to be at least 0.336°.
- Require the criteria to hold for three consecutive frame numbers; the three median estimates must share a sign and have a range no greater than 0.15°.
- The aggregation explicitly checks consecutive frame numbers after aligning cameras by frame and sample token.

An accepted estimate is a correction proposal. `accept_rate` is coverage. Direction accuracy, fraction within 0.10°, and MAE are computed only on accepted frames; they must be read together with coverage.

#### Tracker run command

The additional evaluation batch ran scene indices 30–49, all six cameras, and clean / +0.5° / −0.5° synthetic roll conditions. Run inside the prepared container from `/workspace/bevfusion`:

```bash
cd /workspace/bevfusion

cameras=(
  CAM_FRONT CAM_FRONT_RIGHT CAM_FRONT_LEFT
  CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT
)

for i in $(seq 30 49); do
  for camera in "${cameras[@]}"; do
    for injection in 0 0.5 -0.5; do
      SCENE_INDEX=$i \
      CAMERA_CHANNEL=$camera \
      SUPPORT_MODE=search_stable \
      USE_GT_MOVING_FILTER=0 \
      OBJECTIVE_MODE=kernel_sum \
      INJECTED_ROLL_DEG=$injection \
        python3 experiments/calibration_baseline/run_online_roll_tracker.py \
        || exit 1
    done
  done
done
```

This is 360 tracker runs (20 scene indices × 6 cameras × 3 conditions). Scene indices do not map to consecutive scene-name suffixes. The selected indices map to `scene-0031` through `scene-0052` with gaps, as listed in `/workspace/nuscenes/v1.0-trainval/scene.json`. The aggregation must use those actual names rather than assume a continuous suffix range.

#### Results

The gate threshold was developed on scenes 0011–0020. Results from that development batch and the two later evaluation batches are summarized below. Values are equal-scene means; injected-condition accuracy columns are conditional on the gate accepting a correction.

| Cohort | Condition | Clean false-correction rate | Acceptance / coverage | Correct direction among accepted | Within 0.10° among accepted | Accepted MAE |
|---|---|---:|---:|---:|---:|---:|
| Development, scenes 0011–0020 | Clean | 0.0% | 0.0% | — | — | — |
| Development, scenes 0011–0020 | +0.5° injection | — | 46.3% | 100.0% | 60.4% | 0.087° |
| Development, scenes 0011–0020 | −0.5° injection | — | 41.2% | 100.0% | 51.3% | 0.082° |
| Evaluation, scenes 0021–0030 | Clean | 0.0% | 0.0% | — | — | — |
| Evaluation, scenes 0021–0030 | +0.5° injection | — | 35.8% | 100.0% | 66.0% | 0.076° |
| Evaluation, scenes 0021–0030 | −0.5° injection | — | 15.1% | 100.0% | 60.0% | 0.087° |
| Follow-up, indices 30–49 (20 scenes) | Clean | 0.0% | 0.0% | — | — | — |
| Follow-up, indices 30–49 (20 scenes) | +0.5° injection | — | 23.6% | 100.0% | 63.5% | 0.076° |
| Follow-up, indices 30–49 (20 scenes) | −0.5° injection | — | 29.8% | 100.0% | 65.5% | 0.073° |

For the latest batch, the mean camera spread was 0.899° clean, 0.791° for +0.5° injection, and 0.852° for −0.5° injection. The per-scene results include scenes with no accepted corrections; those scenes contribute zero coverage and have undefined conditional accuracy.

The latest aggregation includes a requirement that the three gate frames have consecutive frame numbers. Its results were unchanged to the three decimal places shown compared with the prior aggregation. The output is:

```text
results/calibration_baseline/multicam_consensus_evaluation_scene_indices_30_49.csv
```

#### Interpretation and limitations

The clean false-correction rate was zero in each reported cohort. In the newest 20-scene batch, the gate accepted 23.6% of +0.5° cases and 29.8% of −0.5° cases; every accepted correction had the expected direction. Among accepted corrections, 63.5% and 65.5%, respectively, were within 0.10° of the target. This supports a selective gate under the tested synthetic ±0.5° CAM roll perturbations. It does not show that the gate can correct most frames, handle natural calibration drift, generalize to pitch/yaw, or improve downstream detection. Frame-level confidence intervals have not yet been computed; future uncertainty estimates should resample whole scenes.

### Frozen score-curve gate evaluation

This is a separate experiment from the 0.336° multi-camera consensus gate above. It trains a Ridge regressor on the six cameras' kernel-sum score curves, then accepts a correction only when its predicted magnitude exceeds a frozen cutoff. The evaluation harness applies this saved model unchanged. It tests synthetic clean, +0.5°, and −0.5° roll conditions; it does not test natural calibration drift.

#### Development and gate selection

`score_curve_frozen_gate.py` was added with two modes: `fit` selects and saves one development gate, while `evaluate` applies that saved gate to an explicit scene list and rejects overlap with the saved development scenes. The selected model was fit on 79 scenes using grouped out-of-fold predictions. Its frozen settings are:

| Setting | Value |
|---|---:|
| Objective | `kernel_sum` |
| Ridge alpha | 1000 |
| Output gain | 0.80 |
| Acceptance cutoff | 0.47° |
| Development OOF clean false-scene rate | 3.8% |
| Minimum development OOF coverage across injection signs | 10.2% |
| Mean accepted MAE used for screening | 0.076° |

The model and settings were saved as `results/calibration_baseline/frozen_kernel_sum_gate.joblib` and `frozen_kernel_sum_gate.json`. The fit is development only; these metrics are not final held-out performance.

#### Earlier ten-scene frozen evaluation

An initial separate ten-scene evaluation used the saved gate without refitting:

| Metric | +0.5° injection (target −0.5°) | −0.5° injection (target +0.5°) |
|---|---:|---:|
| Equal-scene coverage | 11.8% | 11.5% |
| Correct direction among accepted scenes | 100% (8 accepting scenes) | 100% (9 accepting scenes) |
| Within 0.10° among accepted scenes | 58.1% | 65.9% |
| Accepted MAE | 0.086° | 0.083° |

There were 0 clean false-trigger scenes out of 10. The one-sided exact 95% upper bound for the clean false-trigger probability was 25.9%. Scene-bootstrap 95% intervals were wide: coverage was 5.4–19.1% for +0.5° and 5.0–19.5% for −0.5°; accepted MAE intervals were 0.058–0.121° and 0.046–0.134°, respectively. This ten-scene batch is a pilot, not adequate evidence for a low clean false-trigger rate or stable coverage.

#### 27-scene frozen-gate evaluation

The frozen gate was applied unchanged to 27 scenes that were not in the 79-scene development set. This is the final calibration evaluation completed for this project phase. Results:

| Metric | +0.5° injection (target −0.5°) | −0.5° injection (target +0.5°) |
|---|---:|---:|
| Mean scene coverage | 3.6% (95% scene-bootstrap CI 1.8–5.8%) | 9.0% (3.4–15.9%) |
| Clean false-trigger scenes | 0/27 | — |
| One-sided exact 95% upper bound after 0/27 clean events | 10.5% | — |
| Correct direction among accepting scenes | 100% (13 scenes) | 100% (15 scenes) |
| Within 0.10° among accepting scenes | 92.9% (85.3–100.0%) | 79.6% (62.0–94.5%) |
| Accepted MAE | 0.036° (0.021–0.054°) | 0.065° (0.039–0.095°) |

The 27-scene results show accurate corrections when the gate accepts, but coverage is below the 10% development target for both injection signs. Zero clean false-trigger scenes in this evaluation is encouraging, but the one-sided 95% upper bound is 10.5%; this does not establish a clean false-trigger probability below 5%. The evidence supports selective correction on tested synthetic roll perturbations, not a generally reliable calibration method.

#### Evaluation command and implementation notes

The tracker uses NumPy, SciPy KDTree, and OpenCV, so the calibration runs reported here used the CPU; `--gpus all` exposes a GPU to the container but does not accelerate this script.

The saved gate was evaluated with:

```bash
cd /workspace/bevfusion
CURVE_OBJECTIVE_MODE=kernel_sum \
CURVE_SCENES=scene-0107,scene-0108,scene-0109,scene-0110,scene-0120,scene-0121,scene-0122,scene-0123,scene-0124,scene-0125 \
FROZEN_GATE_MODE=evaluate \
FROZEN_GATE_TAG=holdout_batch_01 \
python3 experiments/calibration_baseline/score_curve_frozen_gate.py
```

The evaluation used the saved `frozen_kernel_sum_gate.joblib`; fitting again changes the gate and is not a frozen evaluation. The camera-calibration experiments for this project phase are complete at the scope reported here. No further calibration runs are included in these results.

### Relevant experiment files

- `experiments/calibration_baseline/soac_ocamo_baseline_audit.md` — scientific SOAC reference and OCaMo/LTO adaptation audit.
- `experiments/calibration_baseline/check_nuscenes_projection.py` and `check_projection_sequence.py` — nuScenes projection diagnostics.
- `experiments/calibration_baseline/sweep_rotation_score.py` — candidate rotation score sweep.
- `experiments/calibration_baseline/sweep_rotation_recovery.py` — synthetic correction-grid recovery checks.
- `experiments/calibration_baseline/run_online_roll_tracker.py` — finite/ring-range filtering, support modes, and roll-tracking comparisons under robust-Chamfer and kernel-sum objectives.
- `experiments/calibration_baseline/run_online_roll_tracker_deployable.py` — runtime-support, GT-free diagnostic copy.
- `experiments/calibration_baseline/score_curve_groupcv.py`, `score_curve_nested_gate.py`, and `score_curve_frozen_gate.py` — score-curve feature extraction, nested development screening, and frozen-gate fitting/evaluation.
- CSVs are written under `results/calibration_baseline/`; generated plots and results should be kept with their scene/configuration labels.

## Usage

### Prerequisites

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

We also provide a [Dockerfile](docker/Dockerfile) to ease environment setup. To get started with docker, please make sure that `nvidia-docker` is installed on your machine. After that, please execute the following command to build the docker image:

```bash
cd docker && docker build . -t bevfusion
```

We can then run the docker with the following command:

```bash
nvidia-docker run -it -v `pwd`/../data:/dataset --shm-size 16g bevfusion /bin/bash
```

We recommend the users to run data preparation (instructions are available in the next section) outside the docker if possible. Note that the dataset directory should be an absolute path. Within the docker, please run the following command to clone our repo and install custom CUDA extensions:

```bash
cd home && git clone https://github.com/mit-han-lab/bevfusion && cd bevfusion
python setup.py develop
```

You can then create a symbolic link `data` to the `/dataset` directory in the docker.

### Data Preparation

#### nuScenes

Please follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download and preprocess the nuScenes dataset. Please remember to download both detection dataset and the map extension (for BEV map segmentation). After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```

### Evaluation

We also provide instructions for evaluating our pretrained models. Please download the checkpoints using the following script:

```bash
./tools/download_pretrained.sh
```

Then, you will be able to run:

```bash
torchpack dist-run -np [number of gpus] python tools/test.py [config file path] pretrained/[checkpoint name].pth --eval [evaluation type]
```

For example, if you want to evaluate the detection variant of BEVFusion, you can try:

```bash
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
```

While for the segmentation variant of BEVFusion, this command will be helpful:

```bash
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml pretrained/bevfusion-seg.pth --eval map
```

### Training

We provide instructions to reproduce our results on nuScenes.

For example, if you want to train the camera-only variant for object detection, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

For camera-only BEV segmentation model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/camera-bev256d2.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

For LiDAR-only detector, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

For LiDAR-only BEV segmentation model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/lidar-centerpoint-bev128.yaml
```

For BEVFusion detection model, please run:
```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/lidar-only-det.pth
```

For BEVFusion segmentation model, please run:
```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

Note: please run `tools/test.py` separately after training to get the final evaluation metrics.

## Deployment on TensorRT
[CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion): Best practice for TensorRT, which provides INT8 acceleration solutions and achieves 25fps on ORIN.

## FAQs

Q: Can we directly use the info files prepared by mmdetection3d?

A: We recommend re-generating the info files using this codebase since we forked mmdetection3d before their [coordinate system refactoring](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/changelog.md).

## Acknowledgements

BEVFusion is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). It is also greatly inspired by the following outstanding contributions to the open-source community: [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [BEVDet](https://github.com/HuangJunjie2017/BEVDet), [TransFusion](https://github.com/XuyangBai/TransFusion), [CenterPoint](https://github.com/tianweiy/CenterPoint), [MVP](https://github.com/tianweiy/MVP), [FUTR3D](https://arxiv.org/abs/2203.10642), [CVT](https://github.com/bradyz/cross_view_transformers) and [DETR3D](https://github.com/WangYueFt/detr3d).

Please also check out related papers in the camera-only 3D perception community such as [BEVDet4D](https://arxiv.org/abs/2203.17054), [BEVerse](https://arxiv.org/abs/2205.09743), [BEVFormer](https://arxiv.org/abs/2203.17270), [M2BEV](https://arxiv.org/abs/2204.05088), [PETR](https://arxiv.org/abs/2203.05625) and [PETRv2](https://arxiv.org/abs/2206.01256), which might be interesting future extensions to BEVFusion.


## Citation

If BEVFusion is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```bibtex
@inproceedings{liu2022bevfusion,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xingyu and Mao, Huizi and Rus, Daniela and Han, Song},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```
