# Conditional Backfire Metrics in the §6 Symbol System: Docking Our E1/E4 Results

**Date:** 2026-11-07 · **Purpose:** translate our existing conditional analysis into the $B_c / D_c / G_c$ notation of `Robustness_Experiment_math.md` §6, so wangYilia's upcoming T1B conditional analysis plugs directly into the conditional-weighting design. · **Data:** E4 corrected FN decomposition (miss-rate basis, common subset n = 5,712, 116,027 GT boxes) and E1 range-bin recall (custom matching, recall@2m).

## 1. Symbol alignment (§6.1 of the math doc)

The math doc defines everything on **object miss rates** $M_c$:

$$
B_c = M_{\mathrm{LiDAR},c} - M_{\mathrm{Clean},c},\qquad
D_c(\epsilon) = M_{\mathrm{Noisy},c}(\epsilon) - M_{\mathrm{Clean},c},\qquad
G_c(\epsilon) = D_c - B_c = M_{\mathrm{Noisy},c}(\epsilon) - M_{\mathrm{LiDAR},c}
$$

Our E4 corrected FN% **is** $M_c$ (greedy score-ordered matching, center distance < 2 m, score ≥ 0.1, unconditional FN = every unmatched GT box). So no conversion is needed — only relabeling. Note this is deliberately **not** mAP/NDS; per the math doc, the miss-rate protocol and mAP are not interchangeable.

Perturbation magnitude here: $\epsilon$ = k10 ≈ 958 ms camera delay (T1 protocol, current-frame calibration).

## 2. Per-class table at k10 (the "conditional sensitivity score" $s_c$ for weighting)

All values in miss-rate percentage points. $G_c > 0$ ⇔ fusion backfires in group $c$.

| Class c | M_clean | M_k10 | M_LiDAR | B_c | D_c(k10) | **G_c(k10)** | Backfire? |
|---|---|---|---|---|---|---|---|
| traffic_cone | 14.5 | 27.8 | 18.9 | 4.4 | 13.3 | **+8.9** | ✅ strongest |
| motorcycle | 23.0 | 33.1 | 28.3 | 5.3 | 10.1 | **+4.8** | ✅ |
| bicycle | 34.6 | 44.8 | 40.6 | 6.0 | 10.2 | **+4.2** | ✅ |
| pedestrian | 8.9 | 13.0 | 10.0 | 1.1 | 4.1 | **+3.0** | ✅ |
| car | 8.8 | 13.8 | 11.1 | 2.3 | 5.0 | **+2.7** | ✅ |
| construction_vehicle | 52.9 | 57.6 | 55.8 | 2.9 | 4.7 | **+1.8** | ✅ (weak) |
| trailer | 33.6 | 35.1 | 34.5 | 0.9 | 1.5 | **+0.6** | marginal |
| bus | 14.3 | 17.8 | 18.2 | 3.9 | 3.5 | **−0.4** | ❌ |
| barrier | 14.5 | 16.7 | 17.8 | 3.3 | 2.2 | **−1.1** | ❌ |
| truck | 22.5 | 26.7 | 27.9 | 5.4 | 4.2 | **−1.2** | ❌ safest |
| **overall** | **12.8** | **18.0** | **15.5** | **2.7** | **5.2** | **+2.5** | ✅ |

Source: `results/e4_fp_fn_analysis_2026-10-30_en.md` (corrected FN, 2026-11-03). Chart: `chart_gc_docking.png` panel A.

**Two metric-basis discrepancies worth flagging** (miss-rate vs AP don't always agree):

- `construction_vehicle` and `trailer` show small positive $G_c$ here but were *safe* in the E1 AP table (k10 AP ≥ LiDAR-only). AP integrates score ranking; the miss rate counts every unmatched GT. For the weighting design this means: the choice of $M_c$ definition shifts the membership of the "vulnerable group" at the margin. The math doc's protocol-agnostic wording covers both, but the team should fix one primary $M_c$ definition before Stage D weighting experiments.
- The ordering of the top backfire classes (cone > motorcycle > bicycle > pedestrian > car) is identical under both metrics — only the tail differs. Robust conclusion: **small/dynamic classes are the vulnerable group under both definitions.**

## 3. Distance-dependent gap $\hat{s}(d)$ — the §6.5 weighting proxy input

§6.5 proposes $s_i = \hat{s}(d_i)$ on LiDAR calibration features as a first weighting proxy. From E1 range bins (recall@2m converted to miss%, k8 ≈ 700 ms):

| Distance bin | car $G$ | truck $G$ | bus $G$ |
|---|---|---|---|
| 0–10 m | +0.4 | +0.4 | +0.9 |
| 10–20 m | 0.0 | −0.3 | −0.3 |
| 20–30 m | +0.2 | −0.1 | −0.2 |
| 30–40 m | **+1.0** | +0.8 | **−2.7** |
| 40–60 m | **+1.1** | −0.7 | **−1.9** |

Chart: `chart_gc_docking.png` panel B. Reading:

- **For car, $\hat{s}(d)$ grows with distance** — the monotone proxy §6.5 hopes for holds.
- **For truck/bus it does not**: large vehicles are *safer* at long range under temporal misalignment (their long-range miss rate is LiDAR-dominated, and stale camera features can't hurt what they can't help). This directly illustrates the math doc's caveat that "feature distance alone does not establish that it should inherit an object's detection risk": distance-only weighting would down-weight far features, but far features on large vehicles are exactly where fusion stays safe — while far features near small-object regions are where it backfires. **A defensible Stage-D proxy needs at least (distance × object-size/class) two-dimensional conditioning, or region association via predicted boxes** — consistent with the doc's §6.5 warning and its "object-dependent weighting requires observable grouping" requirement.

## 4. What T1B adds (plug-in point for wangYilia's analysis)

Under T1B the aggregate stays above the LiDAR-only floor at every tested delay (~+2.6 pt at ~1 s, mAP basis). The per-class/per-condition question is whether $G_c(\epsilon)$ stays ≤ 0 across groups or goes positive locally even when the aggregate does not — i.e., whether conditional backfire survives ego-geometry correction. If yes, the affected groups + their observable conditions (object motion, relative displacement, visibility, LiDAR support) become the sensitivity scores for Stage D. If no, the conditional-weighting story weakens for the temporal branch and the spatial rotation results (where distance-driven $G_c$ is already documented, §3) carry the weighting design.

Practical requests for the T1B conditional run, mirroring §6/§9 of the math doc:

1. Report $B_c$, $D_c(\epsilon)$, $G_c(\epsilon)$ **on the miss-rate basis** (same matching protocol as E4) alongside any AP-based view, so both metric families stay available.
2. Screen one-dimensional conditions broadly first (distance, speed, relative displacement, visibility/truncation, LiDAR point support), then pick two-dimensional controls.
3. Keep scene-level independence (§11.1): resample scenes, not frames, for uncertainty.
4. Do not select weighting coefficients on the evaluation scenes (§11.2 leakage rule).

## 5. Reproduce

- Numbers: `results/temporal_misalignment/e4_fp_fn_analysis_2026-10-30_en.md` (per-class FN%), `results/backfire_breakdown_analysis_2026-10-30_EN.md` (range recall), both on branch `jhe-sudo/temporal-misalignment`.
- Chart script: `results/make_gc_docking_chart.py`.
- Matching protocol details (greedy, 2 m, score 0.1, unconditional FN): E4 report header.
