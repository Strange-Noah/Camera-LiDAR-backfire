# Scripts — Temporal Misalignment Pipeline

Two independent halves live here:

- **Part A — inference (GPU).** Perturb the inputs of BEVFusion, run full-validation inference, accumulate predictions. Needs `mmdet3d` + BEVFusion and the nuScenes sensor blobs.
- **Part B — analysis (CPU).** Conditional-threshold and FP/FN analyses over the released predictions. Needs only the nuScenes metadata JSON and `numpy` / `pandas`.

Nothing in Part B touches a GPU, images or point clouds.

## Layout

```
scripts/temporal/
├─ setup_env_1..5.sh          environment build, run in order
├─ extract_val.sh             selective val extraction from the 10 trainval blobs
├─ chunk_pipeline.sh          chunked full-val inference (baselines)
├─ pass2.sh / pass3.sh        two disk-driven recovery passes
├─ gen_subtable.py            T1 substitution table (k = 1, 2) walking sample_data.prev
├─ hist_extract.sh            extract historical camera frames to a persistent folder
├─ chain.sh                   stage A → pilot → full T1 run
├─ temporal_pipeline.sh       T1 main loop (k1/k2 misaligned inference)
├─ temporal_pass2.sh          T1 recovery pass for cross-blob samples
├─ cycle_e2 / e5 / e6 / e6b   cycling drivers for the E-series
├─ accum_*.py                 per-round accumulation, deduplicated by token
├─ delay_stats.py             measured delay distribution from the substitution tables
├─ subset_eval.py             paired same-token evaluation with the official devkit
├─ conditional_threshold_t1/  conditional analysis, Part B stage 1
├─ conditional_threshold_t1b/ conditional analysis, T1B extension
└─ conditional_threshold_e6b/ conditional analysis, E6B extension
```

## Part A — Inference

### 1. Environment

`setup_env_1.sh` … `setup_env_5.sh` run in sequence and are resumable; each appends to `/root/autodl-tmp/setup*.log`. They build a conda environment named `bevfusion` containing `torch 2.1.1+cu121`, `torchvision 0.16.1`, `mmcv 2.1.0`, `mmengine 0.10.7`, `mmdet 3.2.0`, `mmdet3d` v1.4.0 (editable, from `open-mmlab/mmdetection3d`), then compile the BEVFusion CUDA ops and download the LiDAR-only and fusion checkpoints.

The five scripts exist because the image ships CUDA 12.8 while the wheels are cu121; segments 2–5 mostly work around that mismatch and around `triton` / `MarkupSafe` pinning.

### 2. Full-validation baselines

| Script | What it does |
| --- | --- |
| `extract_val.sh` | pulls val sample files from each of the 10 blobs by filename pattern |
| `chunk_pipeline.sh` | per blob: extract → filter the pkl by "file actually present" → infer LiDAR-only and fusion → accumulate → delete data |
| `pass2.sh` | recovers samples whose files live in a different blob |
| `pass3.sh` | recovers 585 samples whose LiDAR sweep the official tar stores under `samples/LIDAR_TOP/` |
| `fix_pass3.py`, `pass3_rerun.sh` | repair a chunk after a failed pass, then re-infer |
| `accumulate2.py`, `merge_eval.py` | merge per-round predictions and evaluate the merged set once with the official devkit |

This is a disk-constrained pipeline: the work volume is 50 GB, the blob archive is 548 GB, so nothing can be extracted in full. Every stage is resumable and accumulates by token.

### 3. T1 substitution tables

| Script | What it does |
| --- | --- |
| `gen_subtable.py` | walks `sample_data.prev` k = 1, 2 steps per sample and camera; writes `sub_table.json` |
| `hist_extract.sh` | extracts every historical camera frame (≈70k images) into a persistent `camhist/` |
| `chain.sh` | waits for extraction, runs a pilot blob with a clean identity re-run, then the full 10 blobs |
| `pilot_check.py` | verifies the pilot: identity re-run must match the full baseline per token |

The pilot is what makes the rest trustworthy — it establishes that the chunked pipeline reproduces the baseline when the perturbation is a no-op.

### 4. T1 main loop

| Script | What it does |
| --- | --- |
| `make_kpkl.py` | builds the k1/k2 misaligned pkl from the chunk base pkl and the substitution table |
| `temporal_pipeline.sh` | per blob: assemble chunk → build misaligned pkl → fusion-only inference → accumulate |
| `temporal_pass2.sh` | recovery pass for samples whose files straddle two blobs |
| `tp2_move.py` | moves (not copies) recovered files into the chunk, to stay inside the disk budget |
| `fix_tp2_sweeps.py` | relocates 501 officially mis-packaged sweep files and validates pkl paths |

### 5. E-series drivers

`cycle_e2.py` (naive stale LiDAR), `cycle_e5.py` (ego-motion compensated stale LiDAR), `cycle_e6.py` (joint grid, both naive), `cycle_e6b.py` (joint grid, both ego-consistent). Each cycles over blobs, writes per-round predictions, and calls `accum_e2.py` / `accum_e6b.py` to merge them.

Supporting scripts: `e4_fp_fn_local.py` and `e4_fp_fn_fixed.py` (FP/FN decomposition), `e4_t1b_fn.py` + `e4_t1b_fn_common.py` (T1B FN), `e4_e6b_fn.py` (E6B FN), `gen_condition_table.py` (T1B condition table), `e3_eval.py` (per-class AP dump), `build_tok2blob.py` (token → blob map used by the drivers).

### 6. Evaluation and plotting

`subset_eval.py` is the single evaluation entry point: pass it any set of prediction JSONs and it reports the common token subset plus paired metrics under `detection_cvpr_2019`. `run_e5_eval.sh` and `run_e6_eval.sh` are wrappers. `delay_stats.py` turns the substitution tables into the measured delay labels. `chart_e2.py`, `chart_e6.py`, `fit_e2_decay.py` produce the decay and grid figures.

## Part B — Conditional analysis

These run on CPU, from a folder containing the nuScenes metadata and the downloaded predictions.

### T1 pipeline (stage 1)

```bash
python step0_download.py --root /content/drive/MyDrive/nuscenes   # or: --download with run_all.py
python run_all.py        --root /content/drive/MyDrive/nuscenes
```

Steps: `step1_features.py` (per-box features) → `step2_match.py` (hit labels and false positives) → `step3_crossovers.py` (class and conditional crossovers) → `step4_controls.py` (r dose-response, controls, bootstrap) → `step5_plots.py` (figures) → `step6_benefit_degradation.py` (benefit B vs degradation D). `common.py` holds the constants, bin edges, I/O and statistics.

Expected layout under `--root`:

```
v1.0-trainval/    nuScenes metadata JSON (~0.4 GB)
team_data/        released prediction JSONs
ct_results/       created by the pipeline
```

`requirements.txt`: `numpy`, `pandas`, `matplotlib`, optionally `ijson`.

### T1B and E6B extensions (stage 2)

`conditional_threshold_t1b/run_t1b.py` and `conditional_threshold_e6b/run_e6b.py` reuse `common.py` and `step2_match.py` from the T1 folder, so **the three folders must sit side by side**. Both expect the T1 pipeline to have been run first, since they consume `ct_results/gt_features.csv.gz`, `ego_samples.csv` and `labels/`.

## Remote helpers

`ssh_run.sh` and `scp_pull.sh` wrap `ssh` / `scp` to the GPU box. Both contain a hard-coded host and port and read a local `.ssh-askpass.sh`; edit them before reuse.

## Where this ran

Inference: AutoDL RTX 4090 D, code at `/root/mmdetection3d`, working disk `/root/autodl-tmp`, sensor blobs at `/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval`. Drivers were copied to `/root/` and launched from there, so most scripts write logs to `/root/autodl-tmp/*.log` and expect output prefixes under `/root/autodl-tmp/`.

Analysis: Google Colab CPU runtime, repository checked out under `MyDrive/nuscenes/script/`.

## Known issues

- **`run_t1b.py` and `run_e6b.py` point at the wrong folder.** Both do `sys.path.insert(..., '..', 'conditional_threshold')`, but the T1 folder is named `conditional_threshold_t1`. Either rename the folder back to `conditional_threshold`, or change those two lines. As shipped they raise `ImportError`.
- **Some scripts referenced in the reports are not in this repository**: `gen_subtable_b.py`, `gen_subtable_b2.py`, `make_kpkl_b.py`, `check_t1b_pilot.py`, `calc_t1_k910.py`, `chart_e5.py`, `remote_e3_eval.py`, `remote_fix_pass3.py`. They lived only on the GPU box, so the T1B/T1B2 *substitution side* cannot be rebuilt from here (the analysis side can).
- **Chinese comments** in several files; the pipeline itself is language-neutral.
- **Hard-coded paths** (`/root/...`) and a baked-in SSH host mean these are single-machine drivers, not portable tools.

## Conventions

- Every inference stage is resumable and accumulates by token with a `_done.json` marker; re-running a stage skips finished tokens instead of duplicating.
- Predictions are written as `<prefix>/<tag>/pred_instances_3d/results_nusc.json` and merged into a single cumulative JSON per setting.
- Evaluation always restricts to the intersection of tokens present in every file being compared, so coverage differences never masquerade as accuracy differences.
- Large artifacts (prediction JSONs, extracted blobs, camera history) are deliberately outside version control.
