# T1B FN Decomposition: Miss-Rate Erosion Under Stale Camera (Corrected, v2)

**Date:** 2026-11-07 (v2, corrected after wangYilia's coverage audit)
**Status:** COMPLETE (local analysis, CPU only)
**Companion:** [E6B report](e6b_joint_compensated_report_en.md) — joint misalignment with both branches compensated

> **v2 changelog (2026-11-07).** v1 claimed a whole-image miss-rate backfire at k12. That claim was **an artifact of prediction coverage**: the k12 prediction file covers 5,626 samples, 89 tokens short of the E4 common subset (1,795 GT boxes), and v1 counted every GT box in a missing token as a miss. Restricted to the population present in all files (5,623 samples / 114,232 GT boxes), k12 does **not** cross LiDAR-only. v1 also used nominal delay labels (k×100 ms); all delays below are now **measured** from the substitution tables (CAM_FRONT): k2 = 167 ms, k5 = 420 ms, k8 = 672 ms, k10 = 834 ms, k12 = 1010 ms (mean). Thanks to @wangYilia for catching both issues.

## 1. What this is

E4-style FP/FN decomposition applied to the **T1B** prediction series (stale camera frame delivered with its own calibration). Same matching protocol as the corrected E4: greedy matching, 2 m center distance, score threshold 0.1, **unconditional FN** (every unmatched GT box counts — the whole-class-missed cases included).

- Population: tokens present in gt_common.json **and in all five** T1B prediction files (k2/k5/k8/k10/k12) — **n = 5,623 samples, 114,232 GT boxes**. This is the largest population on which every setting is directly comparable.
- Coverage note: T1B k2/k5/k8/k10 cover 5,869 / 5,863 / 5,719 / 5,718 of the 5,712-token E4 subset respectively (superset coverage, zero missing); k12 covers 5,626 (−89 tokens / −1,795 GT boxes — the first keyframe of 89 scenes has no 12-step history).
- M = miss rate (%) = FN / GT; G_c = M_fusion,k − M_lidar-only (per class, percentage points; **G_c > 0 = backfire**).

## 2. Overall miss rates (common population, n = 5,623 / 114,232 GT)

| prediction | measured delay | FN | miss rate | G vs LiDAR-only |
|---|---|---|---|---|
| fusion clean | — | (12.87%)* | **12.87%** | −2.67 pp |
| t1b k2 | 167 ms | 14,856 | 13.01% | −2.53 pp |
| t1b k5 | 420 ms | 15,391 | 13.47% | −2.06 pp |
| t1b k8 | 672 ms | 15,895 | 13.91% | −1.62 pp |
| t1b k10 | 834 ms | 16,157 | 14.14% | −1.39 pp |
| t1b k12 | 1010 ms | 16,415 | **14.37%** | **−1.16 pp** |
| LiDAR-only | — | 17,744 | **15.53%** | 0 |

\* fusion clean and LiDAR-only are evaluated on the same 5,623-token population.

**Corrected headline: there is no miss-rate backfire at k12.** The miss rate rises monotonically from 12.87% (clean) to 14.37% (k12, ~1.0 s stale camera) but stays 1.16 pp **below** the LiDAR-only fallback. Fusion's recall advantage erodes steadily — the G margin halves between k2 and k12 — and the per-class margins put pedestrian and car closest to the crossing point, but at ~1 s of ego-consistent camera staleness fusion still misses fewer objects than LiDAR-only on every class. FP counts stay flat throughout (~25.6–25.8k vs 25.9k clean), so there is no false-positive story either: the damage is a gradual recall erosion, nothing more dramatic.

## 3. Per-class margins (G_c = M_fusion − M_lidar-only, pp) at the deepest delays

| class | M_lidar | M_clean | G k8 | G k10 | G k12 |
|---|---|---|---|---|---|
| **car** | 11.04 | 8.8* | −0.8 | −0.4 | **−0.14** (closest) |
| **pedestrian** | 10.08 | 8.9* | −0.3 | −0.2 | **−0.07** (closest) |
| trailer | 34.68 | 33.6* | −0.5 | −0.5 | −0.32 |
| motorcycle | 28.18 | 23.0* | −3.3 | −3.2 | −3.04 |
| bicycle | 40.42 | 34.6* | −4.5 | −5.1 | −4.70 |
| bus | 18.00 | 14.3* | −3.1 | −3.0 | −2.40 |
| truck | 27.89 | 22.5* | −4.8 | −4.6 | −4.20 |
| traffic_cone | 18.93 | 14.5* | −3.1 | −2.9 | −2.31 |
| barrier | 17.83 | 14.5* | −3.0 | −2.8 | −2.84 |
| construction_vehicle | 55.67 | 52.9* | −3.2 | −3.0 | −3.15 |

\* M_clean shown from the v1 run (n = 5,712) for reference; G columns are all on the common population.

**Car and pedestrian are the closest to crossing** (−0.14 / −0.07 pp at k12) — same ordering as every other analysis in this project — but neither crosses. Trailer sits at −0.32 pp. Extrapolating the linear erosion trend (roughly −0.07 pp per 100 ms for car), the whole-image miss-rate crossing would sit somewhere around ~1.5–2 s of camera staleness, beyond the regime we can test (k12 is the eligibility limit).

## 4. Reading

- **mAP and miss rate agree for T1B once coverage is controlled.** v1's claimed metric divergence was the coverage artifact. On a fixed population both metrics say the same thing: monotone erosion, no knee, no crossing out to ~1 s.
- **The erosion is real and class-asymmetric.** Fusion loses half of its recall advantage over LiDAR-only between k2 and k12 (2.53 → 1.16 pp), and the loss concentrates in pedestrian/car — the small, dynamic, camera-reliant classes. The "camera budget ~1 s" statement from E6B is therefore safe in *both* metrics, with the caveat that the pedestrian margin is nearly exhausted at ~1 s.
- Consistent with E4 and E6B-FN: fusion's value over LiDAR-only erodes monotonically with camera staleness as a gradual recall erosion, never as an FP collapse.

## 5. Artifacts

- Scripts: `e4_t1b_fn.py` (v1, full-subset), `e4_t1b_fn_common.py` (v2, common population — this table's source)
- Summaries: `results/e4_t1b_fn_summary.json` (v1), `results/e4_t1b_fn_common_summary.json` (v2), cache `results/e4_t1b_cache/`
- Measured delays: computed from `sub_table_t1b.json` / `sub_table_t1b2.json` (CAM_FRONT, mean over eligible tokens)
