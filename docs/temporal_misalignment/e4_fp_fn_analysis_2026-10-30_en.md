# E4 — Error-Type Decomposition of the Camera–LiDAR Backfire (FP vs FN)

**Date:** 2026-10-30 · **Setting:** BEVFusion camera+LiDAR fusion, nuScenes val common subset (n = 5712 samples, 131,527 GT boxes) · **Method:** greedy score-ordered matching, ground-plane center distance < 2 m, predictions with score < 0.1 excluded from matching and counted as low-score noise

## Headline answer

**The backfire is a false-negative (missed-detection) phenomenon, not a false-positive one.**

From clean alignment to k10 (1.0 s camera lag):

| Metric | fusion_clean | k10 | LiDAR-only |
|---|---|---|---|
| False negatives | 12,607 | **17,400 (+4,793, +38%)** | 15,380 |
| False positives (score ≥ 0.1) | 25,907 | 22,997 (−2,910, −11%) | 28,146 |

False positives *decrease* monotonically with misalignment while false negatives grow monotonically. The mechanism is consistent with the fitted asymptote from T1: stale camera features stop adding information, so fusion silently degenerates toward — and for several classes below — the LiDAR-only operating point, rather than producing phantom detections.

## Per-class change in false negatives (clean → k10)

| Class | FN% clean | FN% k10 | LiDAR-only FN% | dFN (count) | dFP (count) |
|---|---|---|---|---|---|
| car | 8.8 | 13.7 | 11.0 | **+2,489** | −637 |
| traffic_cone | 12.8 | **21.3** | 16.0 | **+856** | −1,184 |
| pedestrian | 8.1 | 11.9 | 9.2 | **+839** | −966 |
| truck | 19.1 | 22.3 | 24.4 | +288 | +221 |
| barrier | 12.7 | 14.3 | 15.9 | +242 | −145 |
| motorcycle | 10.9 | 13.0 | 13.1 | +38 | −19 |
| bicycle | 15.2 | 15.8 | 16.5 | +10 | −114 |
| trailer | 25.4 | 26.9 | 25.9 | +33 | −9 |
| bus | 7.6 | 9.0 | 9.8 | +26 | +5 |
| construction_vehicle | 21.3 | 19.4 | 22.1 | −28 | −62 |

Three observations:

1. **The FN increase concentrates exactly in the E1 backfire classes** (car, cone, pedestrian, truck, barrier, motorcycle). Truck/bus/trailer/construction stay safe, matching E1.
2. **For cones, cars and pedestrians the k10 miss rate exceeds the LiDAR-only rate** (cone 21.3% vs 16.0%; car 13.7% vs 11.0%; pedestrian 11.9% vs 9.2%). Misaligned fusion is actively worse than never using the camera at all for these classes — the literal "backfire".
3. **FP suppression is a side effect, not a benefit.** As fusion degenerates, confident-but-wrong camera-assisted detections lose score and drop out, so FP counts fall together with recall. Operators should not read "fewer false alarms" under latency as a robustness win.

## Why this matters for the safety budget

The T1 v2 fit (mAP = 0.6409 + 0.0523·e^(−d/317 ms), R² = 0.999) shows the degradation is a graceful recall loss, not an error explosion: there is no cliff, but every additional millisecond of camera latency permanently converts into missed objects, and the asymptote sits ~0.9 pt below LiDAR-only mAP. The recommended end-to-end safety budget remains **≈ 0.55–0.57 s** (measured crossover 562–568 ms).

## Reproduce

- E3 remote eval (per-class AP k1–k5 + GT dump): `scripts/temporal/e3_eval.py` → `results/backfire_breakdown/breakdown_k15.json` (+ remote-only `gt_common.json`)
- E4 local analysis (zero GPU): `scripts/temporal/e4_fp_fn_local.py` → `results/temporal_misalignment/e4_fp_fn.json`, `results/temporal_misalignment/e4_fp_fn.svg`
