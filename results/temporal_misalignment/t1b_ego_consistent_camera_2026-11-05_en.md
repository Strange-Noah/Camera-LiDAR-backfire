# T1B: Ego-Consistent Delayed Camera — Resolving the Section-16 Calibration Question

**Date:** 2026-11-05
**Status:** COMPLETE — full inference + paired evaluation done
**Model:** BEVFusion camera–LiDAR fusion (official checkpoint), nuScenes val

## 1. Motivation (the open question)

In T1 we replaced the current camera images with the k-th past camera captures
while **keeping the current frame's calibration** (intrinsics and lidar2cam
extrinsics). This treats the delayed image as if it had been captured from the
current ego pose — the wrong-correspondence error mixes ego motion with world
motion. Section 16 of the project brief left open whether the historical frame
should instead bring **its own** calibration (option B: ego-consistent replay).

T1B implements option B: the historical image is delivered together with its own
`cam2img`, `cam2ego`, and `lidar2cam` (rebuilt per capture from
`calibrated_sensor` and `ego_pose` records, exactly matching mmdet3d's keyframe
values, max abs diff 5.7e-05 in a 100-entry sanity check). LiDAR stays current.
The residual camera–LiDAR mismatch is then purely true world/object motion — the
honest simulation of a delayed camera stream with correct per-frame calibration.

## 2. Setup

- Camera delay k ∈ {2, 5, 8} prev-chain steps ≈ 200 / 400 / 700 ms (measured).
- Eligibility: all 6 cams must have a k-th same-scene prev capture:
  5869 / 5863 / 5719 val samples for k = 2 / 5 / 8.
- Historical images: 77,128 captures staged at /root/camhist_b (absolute paths).
- Calibration: per-capture `cam2img` + `lidar2cam = inv(cam2ego) · inv(ego2global_cam) · ego2global_lidar · lidar2ego`.
- Paired eval: common subset over {fusion clean, LiDAR-only, T1 k2/k5/k8, T1B k2/k5/k8}, official nuScenes metrics.

## 3. Results (mAP, paired subset n = 5713)

Baselines: **fusion clean 0.6869**, **LiDAR-only 0.6496**.

| delay | T1 (current calib) | T1B (ego-consistent) | Δ (T1B − T1) | T1B vs clean |
|---|---|---|---|---|
| ~200 ms (k2) | 0.6712 | 0.6854 | **+0.0142** | −0.15 pt |
| ~400 ms (k5) | 0.6557 | 0.6814 | **+0.0257** | −0.55 pt |
| ~700 ms (k8) | 0.6470 | 0.6784 | **+0.0314** | −0.85 pt |

Full metric rows (NDS / mATE / mASE / mAOE / mAVE / mAAE), from the eval log:

| file | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
|---|---|---|---|---|---|---|---|
| fusion clean | 0.6869 | 0.7163 | 0.2783 | 0.2540 | 0.2986 | 0.2572 | 0.1838 |
| LiDAR-only | 0.6496 | 0.6979 | 0.2829 | 0.2516 | 0.2913 | 0.2594 | 0.1836 |
| T1 k2 | 0.6712 | 0.7075 | 0.2876 | 0.2542 | 0.2999 | 0.2540 | 0.1853 |
| **T1B k2** | **0.6854** | 0.7159 | 0.2788 | 0.2548 | 0.2989 | 0.2510 | 0.1847 |
| T1 k5 | 0.6557 | 0.6990 | 0.2871 | 0.2546 | 0.3046 | 0.2559 | 0.1859 |
| **T1B k5** | **0.6814** | 0.7135 | 0.2805 | 0.2567 | 0.3055 | 0.2435 | 0.1856 |
| T1 k8 | 0.6470 | 0.6945 | 0.2877 | 0.2550 | 0.3067 | 0.2552 | 0.1855 |
| **T1B k8** | **0.6784** | 0.7119 | 0.2808 | 0.2568 | 0.3072 | 0.2412 | 0.1873 |

## 4. Interpretation

**Branch 2 confirmed — strongly.** Delivering the historical camera frame with its
own capture-time calibration (T1B) recovers almost the entire degradation seen in
T1 at every tested delay:

- At ~200 ms, T1B (0.6854) is statistically indistinguishable from the clean
  fusion baseline (0.6869, −0.15 pt); T1 had already lost 1.6 pt.
- At ~700 ms, T1B still sits only 0.85 pt below clean, while T1 dropped 4.0 pt
  and **fell below the LiDAR-only baseline** (0.6470 < 0.6496). Under T1B the
  fusion model remains clearly above LiDAR-only at the same delay.

The gap T1B − T1 grows monotonically with delay (+1.4 → +2.6 → +3.1 pt), which is
exactly the signature of ego-motion error: the longer the stale image is treated
as captured from the current pose, the larger the miscalibration.

**Answer to the Section-16 question:** the calibration choice is not a
presentational detail — it decides what phenomenon the experiment measures.
With current-frame calibration (T1), the measured decay is dominated by an
*artificial ego-motion mismatch*; with capture-consistent calibration (T1B),
the model is remarkably robust to delayed images (≤ 0.85 pt loss at 700 ms), and
the residual decay measures *true content staleness* (world/object motion).

**Recommendation for the paper:** report T1B as the primary temporal-misalignment
protocol (it is the honest delayed-stream simulation), and use the T1-vs-T1B gap
as a quantified decomposition: at ~700 ms, ≈3.1 pt of the observed 4.0 pt decay is
attributable to calibration/ego-motion mismatch and ≈0.9 pt to content staleness.
The previously reported "backfire crossing" (fusion below LiDAR-only) disappears
under the ego-consistent protocol.

## 5. Caveats

- T1B delays (200/400/700 ms) are measured on the camera prev chain
  (12 Hz captures, mean-based); per-sample jitter exists.
- Cross-experiment subsets differ from earlier T1/E2 reports (n = 5856 there).
- One val scene (`n015-2018-11-21-19-21-35`) is missing most sweeps in the public
  blobs; mmdet3d's loader silently skips missing sweeps, and all T1B runs
  completed with zero load failures (5719/5719 predictions at k8). Impact, if
  any, is shared identically across T1/T1B/baselines in the paired subset.
- Pilot verification: T1B predictions differ from T1 on 100% of the 5863
  k5-eligible samples (box-level comparison), confirming the calibration swap
  actually reached the model.

## 6. Artifacts

- Prediction files: `preds_temporal/t1b_k{2,5,8}_cumulative.json`
- Drivers: `gen_subtable_b.py`, `make_kpkl_b.py`, `run_t1b.sh`, `run_t1b_eval.sh`, `check_t1b_pilot.py`
