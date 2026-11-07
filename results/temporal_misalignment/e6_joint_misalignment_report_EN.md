# E6: Joint Camera + LiDAR Temporal Misalignment (Full 3x3 Grid)

**Date:** 2026-11-04
**Status:** COMPLETE — inference 100% coverage (zero skips), paired-subset eval done
**Model:** BEVFusion camera–LiDAR fusion (official checkpoint), nuScenes val

## 1. Setup

- Grid: camera delay k_c ∈ {2, 5, 8} (~200 / 500 / 800 ms) × LiDAR delay k_l ∈ {1, 3, 6} (~50 / 151 / 301 ms), 9 cells.
- Camera branch = T1 semantics (replace with k-th past camera frame, calibration kept at current frame).
- LiDAR branch = E2 naive semantics (k-th past sweep as main cloud, poses as-is, no ego-motion compensation).
- Eligibility per cell: 5719–5869 val samples; **all cells reached 100% of their eligible samples** (5869 / 5863 / 5719).
- Paired eval: common subset over all 11 prediction files (fusion clean, LiDAR-only, 9 grid cells), **n = 5719**, official nuScenes detection metrics.

## 2. Results (mAP, paired subset n = 5719)

Baselines: **fusion clean 0.6870**, **LiDAR-only 0.6497**.

| k_l (LiDAR) ↓ \ k_c (camera) → | ~200 ms | ~500 ms | ~800 ms |
|---|---|---|---|
| **~50 ms** | 0.6028 | 0.5823 | 0.5723 |
| **~151 ms** | 0.4410 | 0.4222 | 0.4102 |
| **~301 ms** | 0.2922 | 0.2745 | 0.2656 |

Full metrics per cell (NDS / mATE / mAOE / mAVE):

| cell | NDS | mATE | mAOE | mAVE |
|---|---|---|---|---|
| c2_l1 | 0.6382 | 0.3634 | 0.3629 | 0.4313 |
| c2_l3 | 0.4915 | 0.5847 | 0.5333 | 0.6609 |
| c2_l6 | 0.3846 | 0.7076 | 0.5969 | 0.7650 |
| c5_l1 | 0.6292 | 0.3582 | 0.3628 | 0.4236 |
| c5_l3 | 0.4866 | 0.5578 | 0.5172 | 0.6564 |
| c5_l6 | 0.3744 | 0.7170 | 0.5980 | 0.7680 |
| c8_l1 | 0.6243 | 0.3567 | 0.3632 | 0.4215 |
| c8_l3 | 0.4822 | 0.5499 | 0.5168 | 0.6476 |
| c8_l6 | 0.3713 | 0.7014 | 0.5951 | 0.7709 |

(Baselines: fusion NDS 0.7163 / mAVE 0.2570; LiDAR-only NDS 0.6980 / mAVE 0.2591.)

## 3. Findings

**3.1 LiDAR delay dominates; camera staleness is cheap by comparison.**
At fixed camera delay, going from 50 ms → 301 ms stale LiDAR costs **31–33 mAP points** (e.g. 0.6028 → 0.2922 at k_c=2). At fixed LiDAR delay, going from 200 ms → 800 ms stale camera costs only **~2–3 points** (e.g. 0.6028 → 0.5723 at k_l=1). The heatmap is essentially striped horizontally: rows (LiDAR delay) separate the regimes, columns (camera delay) barely matter.

**3.2 Every joint-misalignment cell backfires below LiDAR-only.**
All 9 cells (0.2656–0.6028) are below the LiDAR-only baseline 0.6497 — including the mildest cell c2_l1 (camera 200 ms + LiDAR 50 ms), which is **4.7 points below LiDAR-only**. This is much earlier than either modality in isolation: camera-only staleness (T1) tolerated ~0.5 s before crossing, and compensated stale LiDAR (E5) stayed above LiDAR-only up to ~75 ms. With naive (uncompensated) stale LiDAR in the mix, fusion falls below LiDAR-only immediately — consistent with E2, where naive k1 (~50 ms) was already below LiDAR-only. The camera staleness then adds a further ~2–3 points on top.

**3.3 Joint misalignment is approximately additive.**
- c2_l1 observed drop vs fusion clean: 8.4 pt (0.6870 → 0.6028). Sum of individual single-modality drops at matching delays (each measured on its own paired subset, n = 5856): E2 naive k1 ≈ 6.1 pt (0.6855 → 0.6247) + T1 camera k2 ≈ 2 pt ≈ **8.1 pt — essentially additive**.
- c8_l6 observed drop: 42.1 pt. E2 naive k6 alone dropped 36.7 pt and T1 camera k8 only ~1.5 pt; sum ≈ 38 pt, observed 42.1 pt — slightly **super-additive** at large delay: once LiDAR geometry collapses, stale camera features get fused onto a corrupted BEV anchor and cost a few extra points instead of barely mattering. (Cross-experiment comparisons here are approximate: different common subsets.)

**3.4 Velocity and orientation errors explode — dynamic-object ghosting signature.**
mAVE rises from 0.257 (clean) to 0.42–0.77 across the grid, and mAOE from 0.30 to 0.52–0.60. When both modalities are stale, moving objects appear at wrong positions with wrong velocities in both branches simultaneously; scale error (mASE) stays nearly flat (0.254 → 0.27), confirming the errors are localization/timing, not size.

## 4. Interpretation and deployment guidance

- BEVFusion's camera branch leans on LiDAR depth/geometry to lift image features into BEV; stale LiDAR corrupts that shared geometric anchor, so LiDAR freshness is the binding constraint. Prioritize LiDAR latency (or apply ego-motion compensation as in E5, which removes most of the small-delay collapse).
- The camera latency budget is comparatively generous: 200 → 800 ms of camera staleness costs only ~2–3 mAP points **as long as LiDAR is current and geometrically consistent**. (Note the within-subset fact that even k_l=1 cells sit below LiDAR-only: that penalty comes from the *naive 50 ms stale LiDAR*, which E2 already showed backfires on its own — it should not be read as camera staleness hurting at 200 ms.)
- Practical rule of thumb from E2/E5/E6 together: with uncompensated pipelines, ANY stale LiDAR in a fusion stack can make fusion worse than using LiDAR alone; with compensation, fusion tolerates several hundred ms of camera staleness while LiDAR must stay within ~75 ms.

## 5. Caveats

- Single official BEVFusion checkpoint; cross-checkpoint robustness not yet tested.
- T1/E2 reference numbers come from a different common subset (n = 5856) than E6 (n = 5719); within-experiment comparisons are exact, cross-experiment ones approximate.
- Camera delay assumes ~100 ms per camera step (T1 semantics); LiDAR ~50 ms per sweep (20 Hz).

## 6. Artifacts

- Raw eval log: `results/remote_pull/e6_subset_eval.out` (this directory) / `results/temporal_misalignment/e6_subset_eval.out` (repo)
- Chart: `chart_e6_grid.png` / `chart_e6_grid.svg`
- Prediction files: `e6_c{2,5,8}_l{1,3,6}_cumulative.json` on the experiment machine (`/root/autodl-tmp/preds_temporal/`)
- Drivers: `cycle_e6.py`, `run_e6_eval.sh`, `subset_eval.py` (repo `scripts/temporal/`)
