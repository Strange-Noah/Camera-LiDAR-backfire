# Camera–LiDAR Fusion Under Temporal Misalignment: Backfire Located, Quantified, Explained
## T1 + E1 final summary — BEVFusion on nuScenes val (n = 5,712 paired)

**TL;DR — four numbers:**

| # | Finding | Value |
|---|---|---|
| 1 | **Backfire crossover** (misaligned fusion falls below LiDAR-only) | **≈ 570 ms** camera latency (exp fit 568 ms; measured k6–k7 bracket 562–586 ms) |
| 2 | **Permanent penalty** (asymptote of mAP(delay), exp fit R²=0.999) | **0.6409 — ~0.9 pt below the LiDAR-only floor**, never recovers |
| 3 | **Class selectivity** (classes below LiDAR-only at k10) | **6 of 10**: cone, motorcycle, car, pedestrian, bicycle, barrier |
| 4 | **Spatial hotspot** (recall gap vs clean) | **30–60 m** — the LiDAR-sparse band where fusion leans on the camera most |

## What we did

**T1** replaced each camera image with the real frame k steps back along the `sample_data.prev` chain (k = 1–10, ≈77–958 ms mean delay), keeping current calibration/poses/LiDAR/GT, and evaluated all 12 settings (clean, k1–k10, LiDAR-only) with the official nuScenes protocol on an identical common subset of 5,712 val samples. **E1** re-analyzed the same 12 prediction files at zero extra inference cost — per-class AP, day/night/intersection scene splits, and range-bin recall@2m.

## The decay is concave, not linear

mAP(delay) loses 1.22 pt per 100 ms at ~100 ms but only 0.15 pt/100 ms at ~900 ms. An exponential-approach model fits best:

```
mAP(d) = 0.6409 + 0.0523 · exp(−d / 317 ms)     R² = 0.999
```

The earlier linear fit (crossover 660 ms, R²=0.87) was rejected: a straight line through concave data crosses the baseline too late and **overestimates the safe budget**.

![crossover](crossover_fit.png)

## Where the backfire happens (E1)

- **Classes**: backfire hits small, dynamic, camera-reliant objects (cone −3.1 pt, motorcycle −2.9 pt, car −1.5 pt, pedestrian −1.3 pt, bicycle −1.2 pt, barrier −0.4 pt below LiDAR-only at k10). LiDAR-dominant large vehicles (truck, bus, trailer, construction vehicle) never backfire. Class-level onset is ~500 ms — **earlier than the aggregate crossover; mean mAP hides local backfires**.
- **Scenes**: day and intersections backfire (intersections earliest, ~500 ms). **Night does not** — when the camera is already unreliable, misalignment is no worse than dropping it.
- **Range**: degradation concentrates at 30–60 m (car recall@2m 40–60 m: clean 0.939 / k8 0.897 / LiDAR-only 0.908); <20 m is untouched.

## The closed loop

Local damage → global permanent offset. Misalignment feeds corrupted camera features exactly where the model needs the camera most (small objects at long range); these class-level backfires do not average out (mAOE stays ~1.6 pt above LiDAR-only at k9/k10), so the aggregate curve settles **permanently below the LiDAR-only floor** — degraded fusion is an attractor, not a transient.

**Practical takeaway**: beyond ~0.55 s of camera latency, a fusion pipeline with stale images is worse than its LiDAR-only fallback, and increasing the delay does not heal it. Graceful-degradation assumptions do not hold for camera–LiDAR fusion; camera dropout/gating mechanisms are the safer design response.

## Artifacts

- Full reports: `docs/temporal_misalignment/report_2026-10-27_{en,zh}.md` · Breakdown: `results/backfire_breakdown/E1_backfire_breakdown_2026-10-30.md`
- Fit details: `results/temporal_misalignment/crossover_fit.{txt,svg}` · Metrics: `results/temporal_misalignment/metrics_summary.json`
- Pipeline code: `scripts/temporal/` (29 files, AutoDL RTX 4090 D, PyTorch 2.1.0/CUDA 12.1, mmdet3d v1.4.0)
- Prediction JSONs (100–114 MB each) not in git by design; archived on AutoDL + local mirror, to be copied to team shared storage per convention.
