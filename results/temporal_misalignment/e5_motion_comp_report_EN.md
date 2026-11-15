# E5: Motion-Compensated Stale LiDAR — Results Summary (FINAL)

**Date**: full k1–k9 inference completed 2026-11-03 14:16 (n = 5,856/setting, zero failures); paired evaluation completed 2026-11-03 15:56
**Executed by**: @heejoe | **Platform**: AutoDL instance (RTX 4090 D, PyTorch 2.1.0 / CUDA 12.1 / python 3.10, conda env `bevfusion`)
**Dataset**: nuScenes v1.0-trainval, val split
**Corresponding task**: Lark group "AI in Visual Computing Team 1" — Run the temporal misalignment experiments (Experiment E5)

---

## 1. Experiment definition (E5: ego-motion-compensated stale LiDAR)

E2 replaced the current LiDAR sweep with the naive historical sweep k steps back, keeping the historical points in the **historical ego frame** (points rigidly wrong wherever the ego has moved). E5 applies the same k-step replacement but **compensates the historical point cloud for ego motion**: points are transformed historical-ego → global → current-ego using the two keyframes' ego poses. Everything else (association, timestamps, camera, GT) is unchanged. This isolates the two mechanisms behind the E2 backfire:

1. **Geometric misregistration** — removed by compensation (static world is re-registered);
2. **Dynamic-object content error** — NOT removable: other traffic actually moved between the two timestamps; compensation re-registers the background but smears/misplaces moving objects.

## 2. Results (paired common subset, n = 5,856, all 21 files evaluated by `subset_eval.py`)

Baselines reproduce exactly: fusion clean **0.6855**, LiDAR-only **0.6480** — same numbers as all previous paired evals.

| k | ~delay (ms) | E2 naive mAP | **E5 compensated mAP** | E5 NDS | E5 mATE | E5 mAOE | E5 mAVE |
|---|---|---|---|---|---|---|---|
| clean | 0 | — | **0.6855** | 0.7155 | 0.2783 | 0.2983 | 0.2578 |
| 1 | 50 | 0.6247 | **0.6686** | 0.6982 | 0.2980 | 0.3054 | 0.3148 |
| 2 | 100 | 0.5394 | **0.6405** | 0.6723 | 0.3313 | 0.3092 | 0.3914 |
| 3 | 151 | 0.4744 | **0.6106** | 0.6452 | 0.3673 | 0.3266 | 0.4487 |
| 4 | 201 | 0.4133 | **0.5813** | 0.6217 | 0.3906 | 0.3372 | 0.4913 |
| 5 | 251 | 0.3606 | **0.5529** | 0.5999 | 0.4059 | 0.3578 | 0.5104 |
| 6 | 301 | 0.3187 | **0.5254** | 0.5815 | 0.4054 | 0.3854 | 0.4972 |
| 7 | 351 | 0.2869 | **0.4961** | 0.5593 | 0.4079 | 0.4234 | 0.4934 |
| 8 | 402 | 0.2448 | **0.4647** | 0.5379 | 0.4094 | 0.4637 | 0.4765 |
| 9 | 452 | 0.1787 | **0.4327** | 0.5145 | 0.4142 | 0.5331 | 0.4976 |
| LiDAR-only | — | — | **0.6480** | 0.6970 | 0.2832 | 0.2915 | 0.2599 |

Raw eval output: `results/temporal_misalignment/e5_subset_eval.out`. Figure: `chart_e5_decay.svg` (E2 vs E5 vs T1 stale-camera curves).

## 3. Conclusions

1. **Compensation removes the immediate geometric shock, but does NOT prevent the backfire.** At k=1 (~50 ms), naive stale LiDAR is already −2.3 pt **below** LiDAR-only (0.6247 vs 0.6480) — the instant E2 backfire. Compensated stale LiDAR at the same delay is +2.1 pt **above** LiDAR-only (0.6686), a +4.4 pt swing. But E5 crosses below LiDAR-only between k=1 and k=2 — the compensated backfire point is only ~75 ms, barely later than naive (<50 ms).
2. **What compensation really changes is the degradation rate, not the crossing.** Naive E2 decays convex/catastrophically to 0.179 at 452 ms (−47 pt vs LiDAR-only). Compensated E5 decays almost perfectly **linearly**: 0.669 → 0.433 (−23.6 pt over 400 ms, ≈ −0.59 pt per 10 ms, with slight acceleration at large k). At 452 ms E5 is still −21.5 pt below LiDAR-only, but +25.4 pt above naive. Stale-LiDAR fusion backfires in both variants; compensation turns collapse into graceful linear decay.
3. **The residual E5 decay is dynamic-content error, and it dominates from ~75 ms.** Since the static background is re-registered, the linear mAP loss tracks moving-object displacement growth. mAOE worsens monotonically (0.305 → 0.533, far above LiDAR-only 0.292 from k=1 onward) — exactly the orientation-smear signature of re-registered backgrounds with misplaced dynamic objects; mATE also degrades past LiDAR-only from k=2. mAVE stays ~0.45–0.50 (vs 0.26 clean) — velocity is destroyed by content error, unlike the T1 stale-camera case where mAVE recovered.
4. **Stale LiDAR hurts fusion much more than stale camera at equal delay.** T1 (stale camera) at ~100 ms: 0.6712 (−1.6 pt vs clean); E5 (compensated stale LiDAR) at 100 ms: 0.6405 (already below LiDAR-only); E2 (naive) at 100 ms: 0.5394. For deployment latency budgets: **camera latency is forgiving up to ~0.55 s; LiDAR latency is unforgiving past ~0.075 s** (fusion would be better off dropping the stale sweep).
5. Design implication: if the fusion pipeline must tolerate LiDAR staleness, ego-motion compensation is mandatory (it halves the damage at 450 ms) — but the only way to stay above the LiDAR-only line is to keep LiDAR staleness under ~75 ms or gate the fusion on recency.

## 4. Engineering notes

- Full k1–k9 inference via the resumable blob-cycling pipeline (`scripts/temporal/cycle_e5.py`, 30 rounds, zero INFER_FAIL/COMP_FAIL); completion verified by per-round accumulation.
- Mean ego-motion displacement observed in logs: k2 ≈ 0.5 m … k6 ≈ 1.6–2.2 m … k9 ≈ 3.4 m — consistent with ~10–20 km/h ego speed over 0.05–0.45 s.
- Paired eval over all 21 prediction files (fusion clean, LiDAR-only, e2_k1..9, e5_k1..9) on the identical 5,856-sample common subset; E2 numbers reproduce the published E2 curve exactly (0.6247 … 0.1787).
- Disk incident & fix (2026-11-03): e2 prediction JSONs are stored gzip-compressed to fit the 50 GB volume; `subset_eval.py` was patched to read `.gz` transparently and to delete transient padded JSONs immediately after loading (a full-disk crash during the first eval attempt destroyed one e2_k1 `.gz`, which was restored from a byte-identical verified local backup).

## 5. Run manifest

- Driver: `scripts/temporal/cycle_e5.py`; remote log `/root/autodl-tmp/cycle_e5.log` (ALL_DONE 2026-11-03 14:16)
- Predictions (remote): `/root/autodl-tmp/preds_temporal/e5_k{1..9}_cumulative.json` (e2 files gzip-compressed: `e2_k{1..9}_cumulative.json.gz`)
- Eval: `scripts/temporal/run_e5_eval.sh` → `subset_eval.py`; remote log `/root/autodl-tmp/e5_subset_eval.log`
- Figure script: `chart_e5.py` (workspace)

## 6. Next

- E6 (joint camera+LiDAR misalignment, 3×3 grid k_c ∈ {2,5,8} × k_l ∈ {1,3,6}) auto-started on E5 completion and is running; paired eval script `scripts/temporal/run_e6_eval.sh` is ready on the remote host.
- Prediction files exceed GitHub's 100 MB limit; share via GitHub Release attachments as before.
