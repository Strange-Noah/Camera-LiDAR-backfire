# T1B Extension: Full Decay Curve to k12 (~1 s) — No Content-Staleness Knee Found

**Date:** 2026-11-06 (inference 2026-11-05 20:44–21:48, controls to 01:00)
**Status:** COMPLETE — T1B k10/k12 + T1 k10/k12 controls + final 12-file paired evaluation
**Model:** BEVFusion camera–LiDAR fusion (official checkpoint), nuScenes val

## 1. What this extension adds

The first T1B report (2026-11-05) covered k ∈ {2,5,8} (≈200/400/700 ms) and showed that
ego-consistent calibration recovers most of T1's decay. The open follow-up was **where the
residual — true content staleness — actually bites**. We extended both protocols to
k = 10 (≈830 ms) and k = 12 (≈1.0 s, eligibility limit: ~10% of val samples lack 12
same-scene prev captures per camera).

Eligibility / coverage: k10 = 5718 samples, k12 = 5626; **zero load failures** in both
T1B runs and both T1 controls; pilot check passed (T1B k10 differs from T1 k10 on >90%
of samples, box level).

## 2. Results — final paired subset over all 12 files, n = 5623

Baselines: **fusion clean 0.6875**, **LiDAR-only 0.6501**.

| k | ~delay | T1 (current calib) | T1B (ego-consistent) | Δ (T1B−T1) | T1B vs clean | T1 vs LiDAR |
|---|---|---|---|---|---|---|
| 2 | 170 ms | 0.6716 | **0.6860** | +0.0144 | −0.15 pt | +2.2 pt |
| 5 | 420 ms | 0.6562 | **0.6820** | +0.0258 | −0.55 pt | +0.6 pt |
| 8 | 700 ms | 0.6476 | **0.6789** | +0.0313 | −0.86 pt | −0.3 pt |
| 10 | 830 ms | 0.6439 | **0.6778** | +0.0339 | −0.97 pt | −0.6 pt |
| 12 | 1000 ms | 0.6395 | **0.6762** | +0.0367 | −1.13 pt | −1.1 pt |

Full metric rows (NDS / mATE / mASE / mAOE / mAVE / mAAE) in `t1b2_subset_eval.out`
(same directory). Per-class AP at k ∈ {2,5,8} is in `per_class_eval.out` (n = 5713
subset): all 10 classes stay above LiDAR-only under T1B at k8, while 5 classes fall
below it under T1.

## 3. Interpretation

1. **No staleness knee up to 1 s.** T1B's loss grows slowly and roughly linearly
   (≈0.12 pt per additional 100 ms between k2 and k12). Even at ~1 s of camera delay,
   fusion loses only 1.13 pt vs clean and remains **+2.6 pt above LiDAR-only**.
   Extrapolating this slope, content staleness alone would not cross the LiDAR-only
   floor until multi-second delays — far outside the project's misalignment regime.
2. **The T1 backfire is confirmed to be a desynchronization phenomenon.** T1 crosses
   the LiDAR-only floor between k5 and k8 (~500–700 ms) and keeps falling
   (−1.1 pt below the floor at 1 s). The calibration-mismatch share of the total decay
   is stable across delays: (T1−clean) = 4.80 pt at k12 of which (T1B−clean) = 1.13 pt
   is true staleness → **≈77% calibration/ego-motion artifact**, consistent with the
   ≈78% estimate at k8 from the first report.
3. **Practical reading.** A fusion pipeline that keeps per-frame calibration
   synchronized with delayed camera frames (pose-aware replay — what any system with
   hardware timestamps and ego pose can do) is essentially immune to camera latency
   in the sub-second range. The dangerous failure mode is the *silent* one: stale
   frames consumed under the nominal-sync assumption (T1), where backfire onsets at
   ~600 ms and grows steadily.

## 4. Caveats

- Delays are mean realized prev-chain offsets (per-sample jitter exists; T1 measured
  medians 150–1000 ms across k2–k12).
- k12 eligibility (5626/6019 = 93.5%) slightly truncates scene heads; the paired
  subset removes any composition bias across the 12 files.
- One val scene (`n015-2018-11-21-19-21-35`) lacks most sweeps in the public blobs;
  the loader silently skips them, impact is shared identically across all 12 files.

## 5. Artifacts

- Predictions: `preds_temporal/t1b_k{10,12}_cumulative.json`, `k{10,12}_cumulative.json`
- Final eval log: `t1b2_subset_eval.out` (this directory)
- Drivers: `gen_subtable_b2.py`, `master_t1b2.sh` / `master_t1b2_cont.sh`
  (symlink-fallback patch to `make_kpkl.py` for cross-device staging)
