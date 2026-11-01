# Handoff — Building Conditional-Threshold Analysis on T1 Results

**For:** Conditional Threshold task (Yilia Wang & user482879) · **From:** @heyz (temporal misalignment, T1 + E1/E3/E4)
**Last updated:** 2026-11-01

## TL;DR

You do **not** need a GPU, CUDA, or matching PyTorch versions for this. All conditional analysis runs on the **prediction JSONs** we already produced — evaluation is pure-CPU (`nuscenes-devkit` + numpy), works on any laptop. Matching our training/inference environment only matters if you later re-run inference (e.g. E2); the full environment fingerprint is in `baseline_full_val_2026-10-26_en.md`.

## What to download

| File | Size | Content |
|---|---|---|
| `fusion_cumulative.json.gz` | ~101 MB | clean fusion predictions, full val 6,019 samples |
| `lidar_cumulative.json.gz` | ~96 MB | LiDAR-only predictions (reference floor) |
| `k{1..10}_cumulative.json.gz` | ~97–109 MB each | T1 misaligned predictions (stale camera, k prev-chain steps back) |
| `gt_common.json.gz` | ~4.6 MB | GT boxes (class/xyz/size/rotation) for the 5,712-sample common subset |
| `metrics_summary.json`, `eval_summary.txt` | <100 KB | all headline numbers, delays, fit parameters |
| `breakdown_k15.json`, `E1_backfire_breakdown_2026-10-30.md` | <100 KB | per-class AP k1–k5; class/scene/range breakdown |

Prediction JSONs exceed GitHub's 100 MB repo-file limit, so they are **not in the repository** — they are distributed as release assets / shared storage (link in the Lark group; the raw files also sit on the AutoDL instance `/root/autodl-tmp/preds{,_temporal}/` while it is running).

## Key facts to build on

- **Paired subset:** every headline number is computed on the intersection of **5,712 samples** (all 12 prediction files cover it). Always compare on this subset.
- **Delays (mean, measured):** k1 77 ms · k2 168 · k3 254 · k4 332 · k5 423 · k6 507 · k7 586 · k8 677 · k9 ~851 (est.) · k10 ~958 (est.).
- **Headline results:** fusion clean 0.6869 → k10 0.6432; LiDAR-only 0.6496; **crossover ≈ 570 ms** (k6–k7); exp fit `mAP = 0.6409 + 0.0523·e^(−d/317 ms)` (R²=0.999), asymptote ~0.9 pt **below** LiDAR-only (permanent penalty).
- **E1:** backfire classes = cone, motorcycle, car, pedestrian, bicycle, barrier (6/10); night does NOT backfire; hotspot 30–60 m.
- **E4:** backfire is FN-driven (+38% clean→k10); FP −11%.
- **Environment offset:** absolute mAP differs ~0.3–0.5 pt across environments (library versions). Compare deltas within one environment, never absolute numbers across environments.

## How to evaluate / analyze

- Official protocol: `scripts/temporal/subset_eval.py` — runs `DetectionEval` (detection_cvpr_2019) on any subset of prediction files, prints the common subset and paired metrics. Requires `nuscenes-devkit` only (CPU).
- Custom matching (FP/FN, range bins, class conditioning): see `scripts/temporal/e4_fp_fn_local.py` and the E1 breakdown doc for the matching conventions (greedy, score-ordered, 2 m ground-plane center distance).
- Scene conditions (day/night/intersection) need the nuScenes metadata JSONs (`v1.0-trainval` tables, free download from nuscenes.org after login); the E1 doc lists the keyword conventions used.

## Gotchas

- Prediction files are large; decompress with `gunzip` before the devkit eval.
- Sample eligibility shrinks with k (scene-first frames have no history) — that is why the paired 5,712 exists; do not compare raw full-set numbers against it.
- k9/k10 delays are camhist-timestamp estimates (~±5 ms vs driver measurement), k1–k8 are driver-measured.
- The 6 tokens missing from the common subset had unrecoverable LiDAR files in the official tars — ignore them.

## Suggested first steps for conditional threshold

1. Reproduce the headline table from `metrics_summary.json` on your machine (sanity check, ~1 h CPU).
2. Condition the per-class crossover on your variable of interest (speed / scene / density) using the E1/E4 conventions.
3. If a condition shifts the crossover by ≫50 ms, that is a publishable conditional-threshold finding; report paired deltas only.
