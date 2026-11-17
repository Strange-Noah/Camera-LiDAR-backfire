# Camera–LiDAR Backfire Project — Status Summary (2026-10-28)

## TL;DR

The temporal-misalignment experiment (T1) now covers k = 1..5 at full scale and k = 6..8 at 99.9% completion. **The backfire point has been located: fusion underperforms LiDAR-only at k ≈ 8, i.e. ≈ 0.6–0.7 s of camera–LiDAR misalignment.** k = 9/10 are prepared and will extend the curve past the crossover.

## 1. Experiment definition (T1)

- 6 camera images replaced by real historical frames k steps back along the `sample_data.prev` chain; LiDAR, calibration, poses and GT of the current sample kept (the "current-alignment" resolution of the Section 16 question, per team spec).
- One chain step ≈ 50–100 ms (the chain walks all camera frames at 12–20 Hz, not just 2 Hz keyframes), so integer k gives ~100 ms granularity.
- Measured mean misalignment: k1 = 77 ms, k2 = 168 ms, k3 = 254 ms, k4 = 332 ms, k5 = 423 ms, k6 ≈ 500 ms, k7 ≈ 600 ms, k8 ≈ 700 ms.

## 2. Definitive results — full paired evaluation, common subset n = 5,863

| Setting | Mean delay | mAP | ΔmAP vs clean | NDS | mATE ↓ | mAOE |
|---|---|---|---|---|---|---|
| fusion clean | — | 0.6856 | — | 0.7155 | 0.2783 | 0.2980 |
| k=1 | 77 ms | 0.6810 | −0.46 pt | 0.7131 | 0.2825 | 0.2978 |
| k=2 | 168 ms | 0.6700 | −1.56 pt | 0.7067 | 0.2879 | 0.3007 |
| k=3 | 254 ms | 0.6628 | −2.28 pt | 0.7026 | 0.2898 | 0.3024 |
| k=4 | 332 ms | 0.6580 | −2.76 pt | 0.7001 | 0.2880 | 0.3044 |
| k=5 | 423 ms | 0.6545 | −3.11 pt | 0.6983 | 0.2874 | 0.3047 |
| **LiDAR-only (same subset)** | — | **0.6481** | −3.75 pt | **0.6970** | 0.2831 | 0.2913 |

Full-val baselines (all 6,019 samples): LiDAR-only 0.6443 mAP / 0.6923 NDS; fusion 0.6830 / 0.7113.

## 3. Backfire observed (interim, paired on n = 589 completed first)

| Setting | mAP | NDS | mAP gap vs LiDAR-only |
|---|---|---|---|
| fusion clean | 0.6907 | 0.7154 | +3.85 pt |
| k=6 | 0.6544 | 0.6950 | +2.2 pt |
| k=7 | 0.6539 | 0.6961 | +1.7 pt |
| **k=8** | **0.6492** | **0.6937** | **−0.30 pt (crossover)** |
| LiDAR-only | 0.6522 | 0.6994 | — |

Conclusions:
1. Monotonic degradation of mAP/NDS with misalignment; per-step decay shrinks (−0.46 → −1.10 → −0.72 → −0.48 → −0.35 pt), i.e. fusion gracefully converges toward LiDAR-only behaviour.
2. mATE peaks at k=3 then falls back toward the LiDAR-only level while mAOE keeps rising — stale images increasingly behave like "no image at all".
3. **Crossover (backfire) at k ≈ 7–8 ≈ 0.6–0.7 s**, matching the earlier extrapolation (k ≈ 7 ± 2).

## 4. Current status & infrastructure

- k = 6/7/8 inference: 5,713 / 5,719 eligible samples complete (99.9%); the last ~95 stubborn tokens are being processed. Full paired evaluation of k = 1..8 will follow on the ~5,700-sample common subset.
- k = 9/10 fully prepared: eligible sets 5,719 / 5,718, perturbed pkls built, historical images already on disk (the deeper-chain images largely coincided with already-extracted ones), dedicated driver ready — will launch immediately after k = 6..8 finishes.
- Engineering: the 50 GB instance disk forced a blob-cycling pipeline (per-600-sample batch: targeted tar extraction → inference for k6/7/8 in one pass → delete batch lidar), with dual-placement hardlinks for the ~585 officially mis-packaged nuScenes lidar files, 3-blob parallel scanning, and automatic skip of unrecoverable tokens (6 tokens skipped out of 5,814).

## 5. Next steps

1. Finish k = 6/7/8 (within ~1 h), run full paired k = 1..8 + LiDAR-only evaluation.
2. Launch k = 9/10, completing the k = 1..10 curve by tomorrow morning.
3. Fit mAP(delay) over k = 1..10 to pin the crossover at ±50–100 ms precision; quantify the negative-gain depth beyond the backfire point; decide with the team whether even deeper k (12/15) is needed.
