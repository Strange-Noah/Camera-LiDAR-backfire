# E7 — Density vs. Staleness Decomposition of the Stale-LiDAR Loss

**Date:** 2026-11-09
**Branch:** `jhe-sudo/temporal-misalignment`
**Paired evaluation subset:** n = 5,869 nuScenes val tokens (intersection of all eight prediction files)
**Motivation:** Answer the review question on the stale-LiDAR runs (E5 l1/l3/l6): the original implementation uses `sweeps[kl:]` as the sweep list, so a stale main cloud is evaluated with `kl` fewer history sweeps (9 → 8/6/3). Staleness and point-density reduction are therefore mixed. How much of the observed mAP loss is density alone?

---

## 1. Experimental design (2×2 factorial + anchors per staleness level)

For each history-truncation level `kl ∈ {3, 6}` (matching E5 l3 ≈ 151 ms, l6 ≈ 301 ms):

| Variant | Main cloud | History sweeps | Isolates |
|---|---|---|---|
| **D_kl** | current (clean) keyframe | `sweeps[kl:]` (same files as l3/l6) | pure density effect |
| **S_kl** | stale main cloud (ego-compensated, as in E5) | full 9 sweeps | pure staleness effect |
| **L_kl** | stale main cloud | `sweeps[kl:]` | **exact replica** of E5 l3/l6 |

Definitions (mAP deltas relative to clean fusion, all on the same 5,869-token subset):

- Density effect = D_kl − clean
- Staleness effect = S_kl − clean
- Total loss = L_kl − clean
- Interaction = L_kl − D_kl − S_kl + clean  (super-additive if negative)

## 2. Results

| Setting | mAP | NDS | Δ mAP vs clean |
|---|---|---|---|
| Fusion clean (9 history sweeps) | **0.6855** | 0.7154 | — |
| LiDAR-only floor | 0.6479 | 0.6969 | −0.0376 |
| D3 (clean main, 6 history) | 0.6765 | 0.7009 | −0.0090 |
| D6 (clean main, 3 history) | 0.6564 | 0.6671 | −0.0291 |
| S3 (stale main, 9 history) | 0.6626 | 0.6982 | −0.0229 |
| S6 (stale main, 9 history) | 0.6517 | 0.6878 | −0.0338 |
| L3 = E5 l3 replica | **0.6106** | 0.6451 | −0.0749 |
| L6 = E5 l6 replica | **0.5254** | 0.5815 | −0.1601 |

### Decomposition

| Level | Total loss | Density | Staleness | Interaction | Density share | Staleness share | Interaction share |
|---|---|---|---|---|---|---|---|
| kl = 3 (≈151 ms) | −0.0749 | −0.0090 | −0.0229 | **−0.0430** | 12% | 31% | **57%** |
| kl = 6 (≈301 ms) | −0.1601 | −0.0291 | −0.0338 | **−0.0972** | 18% | 21% | **61%** |

(Rows sum exactly: e.g. −0.0090 − 0.0229 − 0.0430 = −0.0749.)

### Replica validation

L3 = 0.6106 and L6 = 0.5254 reproduce the originally published E5 subset numbers (0.6106 / 0.5254 on the 5,856-token E5 subset) **exactly**, confirming the E7 implementation is a faithful replica of the original stale-LiDAR pipeline.

## 3. Key findings

1. **Density alone explains only 12–18% of the stale-LiDAR mAP loss.** A clean main cloud with 6 or 3 history sweeps costs only −0.009 / −0.029 mAP. The density confound in E5 l3/l6 is real but small.
2. **The dominant component is a super-additive interaction (57–61% of the loss).** Stale main cloud + truncated history is far worse than the sum of the two individual perturbations (−0.043 / −0.097 mAP beyond additivity). Mechanistically plausible: with a stale main cloud the model can no longer anchor moving objects at the current timestamp, and the history sweeps — normally its fallback for motion — have also been cut, so errors compound rather than add.
3. **Stale fusion drops far below the LiDAR-only floor.** L3 (0.6106) and especially L6 (0.5254) are well under LiDAR-only (0.6479): a misaligned camera stream actively hurts relative to removing the camera, consistent with the backfire thesis.
4. **Velocity error (mAVE) is a major channel.** Clean mAVE 0.2577 rises to 0.3427 (D3) / 0.5409 (D6) when history is truncated even with a clean main cloud, and to 0.4490 (L3) / 0.4968 (L6) in the stale runs. Velocity estimation depends on the sweep sequence, so history truncation damages it disproportionately; the staleness interaction then amplifies orientation error as well (mAOE 0.2987 → 0.3854 at L6).

## 4. Implications

- **For the E5/E6B conditional analysis:** the occlusion/distance patterns reported in E6B are **not** primarily a point-density artifact. Density alone moves mAP by < 0.01–0.03, while the conditional effects we reported are an order of magnitude larger. Adding a per-sample history-sweep-count covariate would be a cheap robustness check, but it is unlikely to change the conditional conclusions.
- **For the threshold results:** the 50–100 ms LiDAR backfire threshold (from ego-compensated staleness with full history, i.e. the S-variant design) is if anything conservative — the original E5 l-rows mix in an additional interaction penalty. The threshold conclusion stands.
- **Design lesson:** any future "stale sensor" experiment should keep the history length fixed (as in the S-variants) so that staleness and density are not confounded.

## 5. Methods notes

- Data: nuScenes v1.0-trainval, val split, 6,019 keyframe tokens; 5,860 have 9 lidar sweeps, 9 have 8, 150 have 0 (excluded by sweep availability). Paired subset = 5,869 tokens present in all eight prediction files.
- Model: BEVFusion camera–LiDAR (`bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d`), official checkpoint, full-val inference per variant.
- Metrics: official nuScenes detection evaluation (detection_cvpr_2019 config), every setting restricted to the same 5,869-token subset.
- Prediction files: `e7_{D3,D6,S3,S6,L3,L6}_cumulative.json` (remote `preds_temporal/`); per-class AP evaluation in progress.
