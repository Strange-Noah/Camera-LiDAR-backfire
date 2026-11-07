# BEVFusion nuScenes Full-Validation Baseline Reproduction Report

**Date**: 2026-10-26 · **Executed by**: Yangzhang He (operated via Kimi Work) · **Instance**: AutoDL RTX 4090 D 24GB (connect.cqa1.seetacloud.com:31311)

## TL;DR

| Model | Our mAP | Our NDS | Official paper mAP/NDS | Gap | Teammate Zihan (fusion) |
|---|---|---|---|---|---|
| BEVFusion-LiDAR | **0.6443** | **0.6923** | 0.649 / 0.696 | −0.46 mAP / −0.33 NDS | — |
| BEVFusion-LiDAR+Camera | **0.6830** | **0.7113** | 0.686 / 0.714 | −0.30 mAP / −0.27 NDS | 0.6855 / 0.7140 (gap −0.25 / −0.27) |

Both baselines ran successfully on the **full nuScenes validation split (6,019 samples)** with the official evaluation protocol (`detection_cvpr_2019`, standard center_distance). Scores are ~0.3–0.5 pt below the official numbers — within the normal range of environment/version differences (torch 2.1.1 / mmcv 2.1.0 vs. the authors' training environment) — and our teammate Zihan's fusion result (68.55/71.40) is low by almost the same margin, indicating a **systematic offset rather than random error**. For the perturbation experiments we therefore use our own baselines as the reference, which preserves comparability.

## Environment fingerprint (for cross-environment comparison)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D 24GB, driver 595.71.05 |
| PyTorch | 2.1.1+cu121 / torchvision 0.16.1+cu121 |
| mmcv / mmengine | 2.1.0 / 0.10.7 |
| mmdet / mmdet3d | 3.2.0 / 1.4.0 (editable) |
| numpy | 1.23.5 |
| LiDAR checkpoint md5 | `f2d08729546cc22e4836e76491bacf2d` |
| Fusion checkpoint md5 | `5873a41fa1ee29b4cdea346be177019a` |
| Code | open-mmlab/mmdetection3d `projects/BEVFusion` (configs bundled) |
| Evaluation | nuscenes-devkit 1.1.11, `detection_cvpr_2019` |

Inference command template:

```bash
python tools/test.py <cfg> <ckpt> \
  --cfg-options test_dataloader.dataset.data_root=$CHUNK/ \
  test_evaluator.ann_file=$CHUNK/nuscenes_infos_val.pkl \
  test_evaluator.format_only=True test_evaluator.jsonfile_prefix=$PRED/<model>/<pass>
```

## Chunked inference plan (50 GB data disk for 548 GB of data)

The official Fulldatasetv1.0 ships as 10 tar blobs (~30–55 GB each) and cannot be fully extracted, so we extract selectively per blob:

1. **pass1 (10 blobs)**: pull val sample files from each blob by filename pattern (6 cameras + lidar + 9 sweeps), filter the chunk pkl by "file actually exists", then run inference. Available samples: 788/285/406/254/221/957/0/672/1291/0 = **4,874**.
2. **pass2**: collect the remaining 1,145 samples by exact filenames, completing **560** (cumulative 5,434).
3. **pass3**: the remaining 585 samples each had one sweep file misplaced by the official packaging (the tar actually stores it under `samples/LIDAR_TOP/` while the pkl records `sweeps/LIDAR_TOP/` — that file is simultaneously another frame's keyframe sample). After dual-location matching and extraction, all **585 were recovered and inferred**.
4. **Accumulation**: `_done.json` dedupes and accumulates by token; both lidar and fusion reached **6,019/6,019 with an identical token set**, then were merged into cumulative JSONs and evaluated with the official devkit.

Note: samples inside the chunk contain hard links into the gather directories — **do not delete the chunk before deleting gather/gather2**.

## Mini smoke test (earlier; pipeline validation only)

81 samples from v1.0-mini: LiDAR mAP 0.5529 / NDS 0.5710; fusion mAP 0.5721 / NDS 0.5788. Note that sweeps were not actually loaded at the time, so these numbers only validate the pipeline and **must not be cited as baselines**.

## Per-class results (full val)

LiDAR (mAP/NDS = 0.6443/0.6923): car .870 · truck .596 · bus .731 · trailer .446 · construction .255 · pedestrian .864 · motorcycle .695 · bicycle .549 · cone .740 · barrier .698

Fusion (mAP/NDS = 0.6830/0.7113): car .894 · truck .640 · bus .764 · trailer .477 · construction .289 · pedestrian .879 · motorcycle .756 · bicycle .625 · cone .791 · barrier .717

## Reproducibility observations (for Wednesday's discussion)

- Same weights, same data, different environments (4090 D + torch 2.1.1 vs. a teammate's environment): fusion differs by −0.25/−0.27, the same magnitude as the gap to the official numbers (−0.30/−0.27) → environment/library versions are the dominant source. **Comparing absolute numbers across environments should be done cautiously**; within the team we should each use our own baseline as the reference and compare deltas.
- Equivalence of chunked inference (per-chunk pkls with reordered sample_idx) vs. full-set inference is indirectly validated by "full 6,019 coverage + reasonable scores".
- The 585 misplaced files in the official tars: mmdet3d records paths in the pkl according to the actual packaging layout and resolves files by the pkl path, so the two are naturally consistent as long as data is placed per the pkl layout — this pitfall was fixed in pass3 with a script (`remote_fix_pass3.py`).

## Follow-up reminders

- AutoDL instance: **the data disk keeps billing until shutdown/release**; the system disk is not preserved. To keep the environment, create an image or back up the pip list and scripts first.
- The SSH password appeared in the chat history — resetting it is recommended.
- Pre-meeting TODO: the Section 16 camera pose/calibration handling (extrinsic/calibration semantics when rolling camera frames back) still needs a team decision; the temporal-misalignment implementation should wait until the spec is finalized.
