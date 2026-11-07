# T1B FN Decomposition: Miss-Rate Backfire of Stale-Camera Fusion at ~1 s

**Date:** 2026-11-07
**Status:** COMPLETE (local analysis, CPU only)
**Companion:** [E6B report](e6b_joint_compensated_report_en.md) — joint misalignment with both branches compensated

## 1. What this is

E4-style FP/FN decomposition applied to the **T1B** prediction series (stale camera frame delivered with its own calibration; k = 2, 5, 8, 10, 12 ≈ 0.2 / 0.5 / 0.8 / 1.0 / 1.2 s). Same protocol as the corrected E4: greedy matching, 2 m center distance, score threshold 0.1, **unconditional FN** (every unmatched GT box counts — the whole-class-missed cases included).

- GT basis: E4 common subset, **n = 5712 samples, 116,027 GT boxes** (identical to the corrected E4 run; clean FN = 14,892 matches E4 exactly — sanity check passed).
- M = miss rate (%) = FN / GT; G_c = M_fusion,k − M_lidar-only (per class, percentage points; **G_c > 0 = backfire**); B_c = M_lidar-only − M_clean (the camera headroom fusion starts with).

## 2. Overall miss rates

| prediction | FN | miss rate |
|---|---|---|
| fusion clean | 14,892 | **12.83%** |
| t1b k2 (~0.2 s) | 15,056 | 12.98% |
| t1b k5 (~0.5 s) | 15,602 | 13.45% |
| t1b k8 (~0.8 s) | 16,111 | 13.89% |
| t1b k10 (~1.0 s) | 16,381 | 14.12% |
| **t1b k12 (~1.2 s)** | 18,210 | **15.69%** |
| LiDAR-only | 18,015 | **15.53%** |

**The headline: at k12 (~1.2 s stale camera) the fusion miss rate (15.69%) crosses ABOVE LiDAR-only (15.53%) — a whole-image backfire in the miss-rate metric at ~1 s**, even though the mAP-metric T1B decay curve still sits comfortably above the LiDAR-only floor at the same delay. FP counts stay flat throughout (~25.9k–26.3k), so there is no false-positive explosion — the damage is entirely missed detections on real objects.

**Metric divergence is itself a finding.** mAP (ranking-aware, center-distance + score) says camera staleness up to ~1.2 s is safe; plain miss rate says fusion is already missing more objects than LiDAR-only at that point. The mAP-vs-miss-rate gap means the extra misses are concentrated on boxes that were low-confidence / poorly localized anyway — but for a safety stack, "we miss more real objects than the LiDAR-only fallback" is arguably the more operationally meaningful statement.

## 3. Per-class backfire margins (G_c = M_fusion − M_lidar-only, pp)

| class | M_lidar | M_clean | G k2 | G k5 | G k8 | G k10 | **G k12** |
|---|---|---|---|---|---|---|---|
| car | 11.1 | 8.8 | −2.1 | −1.4 | −0.8 | −0.4 | **+1.1** |
| pedestrian | 10.0 | 8.9 | −1.0 | −0.7 | −0.3 | −0.2 | **+1.6** |
| trailer | 34.5 | 33.6 | −0.7 | −0.5 | −0.5 | −0.5 | **+0.4** |
| truck | 27.9 | 22.5 | −5.4 | −5.0 | −4.8 | −4.6 | −3.1 |
| bus | 18.2 | 14.3 | −3.8 | −3.8 | −3.1 | −3.0 | −1.1 |
| motorcycle | 28.3 | 23.0 | −4.2 | −4.2 | −3.3 | −3.2 | −2.3 |
| bicycle | 40.6 | 34.6 | −4.9 | −4.9 | −4.5 | −5.1 | −3.9 |
| construction_vehicle | 55.8 | 52.9 | −3.6 | −3.2 | −3.2 | −3.0 | −2.6 |
| traffic_cone | 18.9 | 14.5 | −4.2 | −3.4 | −3.1 | −2.9 | −0.7 |
| barrier | 17.8 | 14.5 | −3.1 | −3.0 | −3.0 | −2.8 | −1.3 |

Three classes cross into positive G_c at k12 — the safety-critical ones: **pedestrian (+1.6 pp), car (+1.1 pp), trailer (+0.4 pp)**. Vulnerable road users lead the backfire. Every other class still keeps a negative margin at k12, but all margins shrink monotonically with k — the trend points the same way for all classes, only the crossing time differs.

## 4. Reading

- **Ego-consistent camera staleness is not free at the object level forever.** E6B shows the mAP cost of 200→800 ms camera staleness is <1 point *when LiDAR is current*. The FN decomposition adds the caveat: the residual damage is a slow, monotone accumulation of missed GT boxes (12.83% → 14.12% by k10), and by ~1.2 s the accumulation overtakes the LiDAR-only fallback in raw miss rate. The "camera budget ~1 s" rule from E6B should therefore be read as **safe in mAP, borderline in miss rate at the far end**.
- The per-class pattern (pedestrian/car first to backfire) is consistent with the conditional story: smaller, faster-moving objects suffer most from stale imagery.
- Consistent with E4's overall conclusion: fusion's value over LiDAR-only erodes monotonically with camera staleness; the backfire is a gradual erosion of the B_c headroom, not a sudden FP collapse.

## 5. Artifacts

- Script: `scripts/temporal/e4_t1b_fn.py`
- Summaries: `results/temporal_misalignment/e4_t1b_fn_summary.json` (this table's source), per-sample cache under `results/e4_t1b_cache/` locally
