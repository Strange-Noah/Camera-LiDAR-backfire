# E6B: Joint Camera + LiDAR Misalignment with Ego-Motion Compensation on BOTH Branches (Full 3x3 Grid)

**Date:** 2026-11-07
**Status:** COMPLETE — inference 100% coverage (zero skips), paired-subset eval done
**Model:** BEVFusion camera–LiDAR fusion (official checkpoint), nuScenes val

## 1. Setup

- Grid: camera delay k_c ∈ {2, 5, 8} (~200 / 500 / 800 ms) × LiDAR delay k_l ∈ {1, 3, 6} (~50 / 151 / 301 ms), 9 cells.
- **Camera branch = T1B semantics**: replace with the k-th past camera frame **and deliver it with its own (stale) calibration** (ego pose + intrinsics at the past frame), extracted on demand from the image blob via hard links. This is the geometrically consistent delivery mode — the stale image is exactly what the sensor produced at that past timestamp.
- **LiDAR branch = E5 semantics**: k-th past sweep as main cloud **with ego-motion compensation** applied to the current-frame pose (transformed to the stale sweep's coordinates), written as the COMP__ main cloud.
- Eligibility per camera column: 5869 / 5863 / 5719 val samples; **all cells reached 100% of their eligible samples** (zero COMP_FAIL / INFER_FAIL / skips over 30 full sweeps).
- Paired eval: common subset over all 11 prediction files (fusion clean, LiDAR-only, 9 grid cells), **n = 5719**, official nuScenes detection metrics.

## 2. Results (mAP, paired subset n = 5719)

Baselines: **fusion clean 0.6870**, **LiDAR-only 0.6497** (the backfire floor).

| k_l (LiDAR) ↓ \ k_c (camera) → | ~200 ms | ~500 ms | ~800 ms |
|---|---|---|---|
| **~50 ms** | 0.6671 | 0.6634 | 0.6604 |
| **~151 ms** | 0.6070 | 0.6042 | 0.6018 |
| **~301 ms** | 0.5242 | 0.5189 | 0.5161 |

Full metrics per cell (NDS / mATE / mAOE / mAVE):

| cell | NDS | mATE | mAOE | mAVE |
|---|---|---|---|---|
| c2_l1 | 0.6972 | 0.2995 | 0.3051 | 0.3142 |
| c2_l3 | 0.6416 | 0.3714 | 0.3270 | 0.4635 |
| c2_l6 | 0.5791 | 0.4079 | 0.3902 | 0.5099 |
| c5_l1 | 0.6962 | 0.3000 | 0.3096 | 0.2972 |
| c5_l3 | 0.6418 | 0.3709 | 0.3288 | 0.4421 |
| c5_l6 | 0.5759 | 0.4117 | 0.3874 | 0.5117 |
| c8_l1 | 0.6955 | 0.2999 | 0.3092 | 0.2896 |
| c8_l3 | 0.6422 | 0.3687 | 0.3285 | 0.4256 |
| c8_l6 | 0.5759 | 0.4092 | 0.3888 | 0.4948 |

(Baselines: fusion NDS 0.7163 / mAVE 0.2570; LiDAR-only NDS 0.6980 / mAVE 0.2591.)

## 3. Findings

**3.1 The camera column is now nearly free: ego-consistent delivery removes most of the camera staleness cost.**
At fixed LiDAR delay, going from 200 ms → 800 ms stale camera costs only **0.5–0.7 mAP points** across all three rows (e.g. 0.6671 → 0.6604 at k_l=1). For comparison, the same camera-delay range cost **~3 points in E6** (naive delivery, calibration kept at the current frame: 0.6028 → 0.5723). Delivering the stale camera frame with its own calibration (T1B) roughly **halves the remaining camera penalty** after ego-motion consistency is established. BEVFusion's camera branch tolerates ego-consistent image staleness up to ~800 ms essentially for free — as long as the images and their calibration are internally consistent.

**3.2 The LiDAR row still dominates — compensation does NOT make stale LiDAR safe.**
The heatmap remains striped horizontally, and the regime boundaries barely move vs the uncompensated case:
- **l1 row (~50 ms stale, compensated): 0.660–0.667, still above the LiDAR-only floor** (+1.1 to +1.7 points).
- **l3 row (~151 ms): 0.602–0.607, ALL THREE cells backfire ~4.4–4.8 points below LiDAR-only (0.6497).**
- l6 row (~301 ms): 0.516–0.524, ~13 points below floor.

The **"compensate both branches and fusion is immune" hypothesis is rejected.** Ego-motion compensation fixes the *static* geometry of a stale LiDAR sweep, but the stale sweep still contains **dynamic content (moving objects) at wrong positions**; fusing it corrupts the shared BEV anchor that the camera branch depends on. This is fully consistent with E5 single-modality: compensated stale LiDAR alone backfired already at ~75 ms (E5 k3 ≈ 0.6106, n = 5856 — remarkably close to the E6B l3 row ~0.602–0.607 measured on n = 5719).

**3.3 Compensation massively rescues the naive collapse, but cannot push it above the floor.**
Same cell, naive (E6) vs compensated (E6B): c2_l3 = 0.4410 → **0.6070 (+16.6 points)**; the entire l3 row rises from ~0.41–0.44 to ~0.60–0.61. Compensation recovers a huge amount of mAP — it just recovers it to a level that is *still below LiDAR-only*. I.e. stale-LiDAR fusion with compensation is much better than naive stale-LiDAR fusion, but still worse than simply dropping the stale LiDAR.

**3.4 Error signature stays dynamic-object localization.**
mAVE rises from 0.257 (clean) to 0.43–0.51 on the l3/l6 rows (vs 0.66–0.77 for naive E6 — compensation roughly halves velocity error too); mASE stays flat (~0.254–0.263), confirming the residual damage is timing/localization on moving objects, not size estimation.

## 4. Interpretation and deployment guidance (closed-loop design law)

Bringing E2 (naive LiDAR), E5 (compensated LiDAR, single-modality), E6 (joint naive), E6B (joint compensated) and T1/T1B (camera-only) together:

1. **LiDAR freshness is the binding constraint, and ego-motion compensation does not relax it enough.** With compensation, the stale-LiDAR fusion backfire threshold moves from ~50 ms (E2 naive) to somewhere between 50 ms and ~151 ms (E6B: l1 above floor, l3 below). Compensation is worth ~100 ms of extra LiDAR budget — helpful, not sufficient.
2. **The camera branch is robust once delivered ego-consistently.** Geometrically consistent stale camera frames (own calibration) cost <1 point up to ~800 ms. The camera latency budget is therefore ~an order of magnitude more generous than the LiDAR one.
3. **Practical rule:** if the LiDAR sweep is older than ~50–75 ms, **drop it and fall back to LiDAR-only** (or run a per-object temporal prediction / correction stage to refresh the dynamic content — directly connects to the team's robustness direction) rather than fusing a stale sweep, *regardless of compensation*. If LiDAR is current, camera frames up to ~1 s can be used as-is provided they arrive with their own calibration.

## 5. Caveats

- Single official BEVFusion checkpoint; cross-checkpoint robustness not yet tested.
- E5 reference numbers come from a different common subset (n = 5856) than E6B (n = 5719); within-experiment comparisons are exact, cross-experiment ones approximate.
- Delay mapping: camera ≈100 ms per step (keyframe), LiDAR ≈50 ms per sweep (20 Hz); l1/l3/l6 ≈ 50/151/301 ms.
- Compensation quality depends on the nuScenes ego-pose accuracy; a real deployment with noisier odometry would see a smaller compensation benefit.

## 6. Artifacts

- Raw eval log: `results/temporal_misalignment/e6b_subset_eval.out`
- Drivers: `scripts/temporal/cycle_e6b.py`, `scripts/temporal/accum_e6b.py`
- Prediction files: on the experiment machine (`/root/autodl-tmp/preds_temporal/e6b_c{2,5,8}_l{1,3,6}_cumulative.json`)
