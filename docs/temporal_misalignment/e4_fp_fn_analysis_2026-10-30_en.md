# E4 — Error-Type Decomposition of the Camera–LiDAR Backfire (FP vs FN)

**Date:** 2026-10-30 · **FN definition corrected 2026-11-03** (see note below) · **Setting:** BEVFusion camera+LiDAR fusion, nuScenes val common subset (n = 5712 samples, 131,527 GT boxes) · **Method:** greedy score-ordered matching, ground-plane center distance < 2 m, predictions with score < 0.1 excluded from matching and counted as low-score noise; **FN = every unmatched GT box** (unconditional)

> **Correction note (2026-11-03).** The original script only counted a class's FN when the sample contained at least one same-class prediction ≥ 0.1, silently skipping GT boxes in samples where the model missed the whole class. That selection effect grows as the model degrades and biased the FN growth downward (reported +38%, corrected +40.4%). FP counts are unaffected. Credit to @heyz for catching this. The corrected numbers below supersede the original table; all headline conclusions are unchanged and the class-level backfire is strictly stronger.

## Headline answer

**The backfire is a false-negative (missed-detection) phenomenon, not a false-positive one.**

From clean alignment to k10 (1.0 s camera lag), corrected FN definition:

| Metric | fusion_clean | k10 | LiDAR-only |
|---|---|---|---|
| False negatives | 14,892 | **20,903 (+6,011, +40.4%)** | 18,015 |
| False positives (score ≥ 0.1) | 25,907 | 22,997 (−2,910, −11%) | 28,146 |

False positives *decrease* monotonically with misalignment while false negatives grow monotonically. The mechanism is consistent with the fitted asymptote from T1: stale camera features stop adding information, so fusion silently degenerates toward — and for the camera-reliant classes below — the LiDAR-only operating point, rather than producing phantom detections.

## Per-class change in false negatives (clean → k10, corrected FN)

| Class | FN% clean | FN% k10 | LiDAR-only FN% | dFN (count) | dFP (count) |
|---|---|---|---|---|---|
| car | 8.8 | 13.8 | 11.1 | **+2,530** | −637 |
| traffic_cone | 14.5 | **27.8** | 18.9 | **+1,333** | −1,184 |
| pedestrian | 8.9 | 13.0 | 10.0 | **+923** | −966 |
| truck | 22.5 | 26.7 | 27.9 | +371 | +221 |
| barrier | 14.5 | 16.7 | 17.8 | +328 | −145 |
| motorcycle | 23.0 | 33.1 | 28.3 | +179 | −19 |
| bicycle | 34.6 | 44.8 | 40.6 | +178 | −114 |
| construction_vehicle | 52.9 | 57.6 | 55.8 | +68 | −62 |
| bus | 14.3 | 17.8 | 18.2 | +68 | +5 |
| trailer | 33.6 | 35.1 | 34.5 | +33 | −9 |

Three observations:

1. **The FN increase concentrates exactly in the E1 backfire classes** (car, cone, pedestrian, motorcycle, bicycle; barrier close behind). Truck/bus/trailer stay safe, matching E1.
2. **For all five small/dynamic classes the k10 miss rate exceeds the LiDAR-only rate** — car 13.8% vs 11.1%, cone 27.8% vs 18.9%, pedestrian 13.0% vs 10.0%, motorcycle 33.1% vs 28.3%, bicycle 44.8% vs 40.6%. Misaligned fusion is actively worse than never using the camera at all for these classes — the literal "backfire". (With the old conditional FN definition, motorcycle and bicycle looked marginally *better* than LiDAR-only; correcting the definition removes that artifact.)
3. **FP suppression is a side effect, not a benefit.** As fusion degenerates, confident-but-wrong camera-assisted detections lose score and drop out, so FP counts fall together with recall. Operators should not read "fewer false alarms" under latency as a robustness win.

## Why this matters for the safety budget

The T1 v2 fit (mAP = 0.6409 + 0.0523·e^(−d/317 ms), R² = 0.999) shows the degradation is a graceful recall loss, not an error explosion: there is no cliff, but every additional millisecond of camera latency permanently converts into missed objects, and the asymptote sits ~0.9 pt below LiDAR-only mAP. The recommended end-to-end safety budget remains **≈ 0.55–0.57 s** (measured crossover 562–568 ms).

## Reproduce

- E3 remote eval (per-class AP k1–k5 + GT dump): `scripts/temporal/e3_eval.py` → `results/backfire_breakdown/breakdown_k15.json` (+ remote-only `gt_common.json`)
- E4 local analysis (zero GPU): `scripts/temporal/e4_fp_fn_local.py` (corrected FN definition) / `scripts/temporal/e4_fp_fn_fixed.py` (old-vs-new A/B) → `results/e4_fp_fn.json` / `results/e4_fp_fn_fixed.json`, `results/temporal_misalignment/e4_fp_fn.svg`
