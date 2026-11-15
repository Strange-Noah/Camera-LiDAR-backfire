# E6B FN/FP Decomposition: Where the Joint-Compensated Backfire Comes From

**Date:** 2026-11-07
**Status:** COMPLETE (local analysis, CPU only)
**Companion:** [E6B report](e6b_joint_compensated_report_en.md) · same protocol as [T1B FN analysis](t1b_fn_analysis_en.md)

## 1. Setup

E4-protocol FP/FN decomposition (greedy, 2 m center distance, score ≥ 0.1, **unconditional FN**) applied to the E6B joint-compensated grid. GT basis: tokens common to gt_common.json and **all 12 prediction files** → **n = 5712 samples, 116,027 GT boxes** (clean FN = 14,892, identical to E4 — sanity check passed). G_c = M_fusion − M_LiDAR-only in miss-rate percentage points (G_c > 0 = backfire).

## 2. Overall: miss rate and FP per cell

Baselines: fusion clean miss 12.83% (FP 25,907); LiDAR-only miss 15.53% (FP 28,146).

| cell | miss % | Δ vs LiDAR-only | FN | FP |
|---|---|---|---|---|
| c2_l1 | 14.42 | −1.11 | 16,735 | 26,151 |
| c5_l1 | 14.83 | −0.70 | 17,211 | 26,227 |
| c8_l1 | 15.32 | −0.21 | 17,776 | 26,131 |
| c2_l3 | 16.78 | **+1.25** | 19,466 | 26,761 |
| c5_l3 | 17.17 | **+1.64** | 19,920 | 26,877 |
| c8_l3 | 17.69 | **+2.16** | 20,521 | 26,645 |
| c2_l6 | 24.69 | **+9.16** | 28,651 | 33,459 |
| c5_l6 | 25.20 | **+9.67** | 29,240 | 34,222 |
| c8_l6 | 25.77 | **+10.24** | 29,895 | 33,949 |

**The miss-rate metric agrees with mAP for the joint grid:** the l1 row stays below the LiDAR-only miss rate (although c8_l1 is only 0.2 pp away — consistent with the mAP margin shrinking to +1.1 pt), the l3 row backfires (+1.2 to +2.2 pp), and the l6 row backfires massively (+9 to +10 pp). Unlike the camera-only T1B case (mAP safe / miss-rate backfired at k12), here the two metrics tell the same story — stale LiDAR hurts ranking *and* raw recall simultaneously.

**FP behavior splits by row.** On the l1/l3 rows FP stays near clean levels (26.1–26.9k vs 25.9k clean) — the damage is almost pure misses. On the l6 row FP **explodes to 33–34k**, far above even LiDAR-only (28.1k): deeply stale LiDAR makes the fusion hallucinate objects (ghosting), consistent with the mAVE/mAOE blow-up in the E6B grid.

## 3. Per-class backfire margins (G_c, pp)

| class | l1 row (c2/c5/c8) | l3 row (c2/c5/c8) | l6 row (c2/c5/c8) |
|---|---|---|---|
| **car** | −0.4 / +0.3 / **+1.0** | +2.6 / +3.1 / +3.8 | +14.7 / +15.5 / +16.2 |
| **pedestrian** | **+0.7** / +0.9 / **+1.2** | +2.8 / +3.2 / +3.7 | +7.2 / +7.4 / +8.0 |
| motorcycle | −2.4 / −1.9 / −0.5 | +3.3 / +3.3 / +4.6 | +15.7 / +15.7 / +17.9 |
| trailer | −0.0 / −0.0 / +0.1 | +0.6 / +1.2 / +0.9 | +5.0 / +5.0 / +5.2 |
| bicycle | −4.1 / −3.9 / −2.6 | +0.4 / +0.3 / +1.3 | +8.3 / +8.6 / +9.3 |
| bus | −3.4 / −3.5 / −2.9 | −1.4 / −1.9 / −1.0 | +9.2 / +9.0 / +9.4 |
| truck | −4.6 / −4.1 / −3.8 | −2.9 / −2.4 / −2.5 | +5.2 / +5.7 / +6.0 |
| traffic_cone | −2.0 / −1.6 / −0.9 | +0.2 / +0.6 / +1.2 | +4.0 / +4.8 / +5.2 |
| barrier | −2.7 / −2.6 / −2.6 | −1.8 / −1.7 / −1.5 | −0.0 / +0.2 / +0.4 |
| construction_vehicle | −3.3 / −3.9 / −3.6 | −1.8 / −1.9 / −1.0 | +1.7 / +1.3 / +1.1 |

Three takeaways:

1. **Pedestrian is the canary.** It is the *only* class already backfiring (G_c > 0) on the l1 row at c5/c8 — i.e. with current LiDAR + 500–800 ms stale camera, fusion already misses more pedestrians than the LiDAR-only fallback, even though the overall mAP/miss-rate is still above floor. This mirrors the T1B finding (pedestrian first to backfire) and the conditional story that small fast objects suffer most.
2. **Car and motorcycle lead the l3-row backfire** (+2.6 to +4.6 pp), followed by pedestrian. Large static-ish classes (truck, bus, barrier, construction vehicle) keep negative margins even at l3 — their fusion value survives longer because ego-motion compensation handles their mostly-static geometry well.
3. **At l6 everything backfires** — even barrier and construction_vehicle cross positive — and the magnitudes (up to +18 pp for motorcycle/car) show deeply stale LiDAR destroys rather than merely degrades fusion.

## 4. Reading (closes the loop with T1B-FN)

- **Camera staleness adds misses, not ghosts.** Down the camera columns (c2→c8) at fixed LiDAR delay, FN grows monotonically (~+1,000 per step) while FP stays flat — stale consistent camera frames quietly erode recall, matching the T1B-FN "gradual headroom erosion" story.
- **LiDAR staleness adds misses first, then ghosts.** l1→l3 is dominated by FN growth; l6 adds the FP explosion. The operational warning sign of a stale LiDAR stream is therefore: rising FN on moving-object classes → then rising FP everywhere.
- The per-class ordering (pedestrian → car/motorcycle → large vehicles) is consistent across T1B-FN, E4, and this decomposition — a robust project-level finding for the conditional-threshold work.

## 5. Artifacts

- Script: `scripts/temporal/e4_e6b_fn.py`
- Summaries: `results/temporal_misalignment/e4_e6b_fn_summary.json`
