# E1 — Where does the backfire happen? Class-, scene-, and range-level breakdown

Date: 2026-10-30 · Dataset: nuScenes v1.0-trainval (val split) · Common subset n = 5712 (intersection of all 12 prediction files, identical protocol to the final T1 paired evaluation)

## TL;DR

1. **6 of 10 classes backfire at k10 (~958 ms misalignment)**, i.e. misaligned fusion drops below the LiDAR-only baseline: traffic cone (-3.1 pt), motorcycle (-2.9 pt), car (-1.5 pt), pedestrian (-1.3 pt), bicycle (-1.2 pt), barrier (-0.4 pt). These are small, dynamic, camera-reliant objects.
2. **4 classes never backfire**: truck, bus, trailer, and construction_vehicle stay above LiDAR-only even at k10 (+0.4 ~ +2.6 pt). Large vehicles have enough LiDAR returns; stale camera features do little damage.
3. **Scene split**: backfire occurs in day and at intersections, but **not at night** (k10 still +0.3 pt above LiDAR-only). At night the camera contributes little anyway, so there is little for misalignment to corrupt.
4. **Spatial split**: degradation concentrates at **30–60 m**; close range (<20 m) is essentially unaffected. E.g. car recall@2m at 40–60 m: clean 0.939 / k8 0.897 / LiDAR-only 0.908 — k8 is already below LiDAR-only there.

## 1. Per-class AP (mean over the 4 distance thresholds, common subset n=5712)

| Class | clean | k6 | k7 | k8 | k9 | k10 | LiDAR-only | k10 − LiDAR | backfire? |
|---|---|---|---|---|---|---|---|---|---|
| car | 0.8945 | 0.8662 | 0.8642 | 0.8609 | 0.8594 | 0.8577 | 0.8724 | **-0.0147** | yes |
| truck | 0.6423 | 0.6147 | 0.6128 | 0.6082 | 0.6047 | 0.6035 | 0.5998 | +0.0037 | no |
| bus | 0.7668 | 0.7509 | 0.7483 | 0.7487 | 0.7453 | 0.7411 | 0.7363 | +0.0048 | no |
| trailer | 0.4873 | 0.4877 | 0.4826 | 0.4833 | 0.4810 | 0.4820 | 0.4556 | +0.0264 | no |
| construction_vehicle | 0.2955 | 0.2686 | 0.2687 | 0.2664 | 0.2670 | 0.2652 | 0.2610 | +0.0042 | no |
| pedestrian | 0.8811 | 0.8586 | 0.8571 | 0.8557 | 0.8540 | 0.8537 | 0.8669 | **-0.0132** | yes |
| motorcycle | 0.7615 | 0.6886 | 0.6848 | 0.6817 | 0.6775 | 0.6761 | 0.7053 | **-0.0292** | yes |
| bicycle | 0.6360 | 0.5573 | 0.5548 | 0.5537 | 0.5499 | 0.5466 | 0.5584 | **-0.0118** | yes |
| traffic_cone | 0.7901 | 0.7252 | 0.7213 | 0.7168 | 0.7149 | 0.7117 | 0.7423 | **-0.0306** | yes |
| barrier | 0.7143 | 0.6968 | 0.6930 | 0.6943 | 0.6941 | 0.6939 | 0.6979 | **-0.0039** | yes |

Key observations:

- **Backfire is class-selective.** The 6 backfiring classes are camera-dependent and small/dynamic; the 4 "large vehicle" classes remain safe because LiDAR geometry alone sustains them.
- **Onset is early.** At k6 (~500 ms) car (0.8662 < 0.8724), motorcycle (0.6886 < 0.7053) and bicycle (0.5573 < 0.5584) are already below LiDAR-only — earlier than the overall mAP crossover (~650 ms). **The aggregate metric hides early, local backfires.**
- Trailer is a special case: clean fusion only gains +3.2 pt over LiDAR-only, and misalignment barely hurts it (k10 0.4820), because LiDAR dominates this class.

## 2. Scene split (mAP within the common subset)

| Scene | n | clean | k6 | k7 | k8 | k9 | k10 | LiDAR-only | backfire? |
|---|---|---|---|---|---|---|---|---|---|
| day | 5146 | 0.6878 | 0.6517 | 0.6491 | 0.6475 | 0.6449 | 0.6434 | 0.6510 | yes (crossover between k6–k7, ~550 ms) |
| night | 566 | 0.4091 | 0.3753 | 0.3722 | 0.3695 | 0.3688 | 0.3654 | 0.3625 | no (k10 still +0.3 pt above) |
| intersection | 1982 | 0.6762 | 0.6393 | 0.6369 | 0.6351 | 0.6344 | 0.6328 | 0.6380 | yes (crossover at ~k6, ~500 ms) |
| other | 3730 | 0.6900 | 0.6552 | 0.6526 | 0.6506 | 0.6475 | 0.6460 | 0.6542 | yes (crossover between k6–k7) |

- **Day**: same trend as the full set, crossover between k6 and k7.
- **Night**: clean fusion gains +4.7 pt over LiDAR-only (0.4091 vs 0.3625); misalignment erases almost all of that gain (k10 only +0.3 pt) but **never falls below LiDAR-only**. Mechanistically: poor nighttime images → small intrinsic camera contribution → stale camera features can neither help nor hurt much. Reassuring safety-side evidence: **under conditions where the camera is already unreliable, temporal misalignment cannot make things worse than simply not using the camera.**
- **Intersections** (dense, dynamic interactions) show the earliest crossover (~k6, ~500 ms).

## 3. Range-bin recall@2m (custom matching, center distance < 2 m)

| bin | car clean | car k8 | car LiDAR | truck clean | truck k8 | truck LiDAR | bus clean | bus k8 | bus LiDAR |
|---|---|---|---|---|---|---|---|---|---|
| 0–10 m | 0.997 | 0.996 | 1.000 | 0.981 | 0.982 | 0.985 | 0.991 | 0.991 | 1.000 |
| 10–20 m | 0.993 | 0.993 | 0.993 | 0.960 | 0.958 | 0.955 | 0.983 | 0.988 | 0.985 |
| 20–30 m | 0.984 | 0.978 | 0.980 | 0.939 | 0.939 | 0.938 | 0.969 | 0.961 | 0.959 |
| 30–40 m | 0.971 | **0.946** | **0.956** | 0.941 | 0.913 | 0.921 | 0.917 | 0.901 | 0.874 |
| 40–60 m | 0.939 | **0.897** | **0.908** | 0.841 | 0.800 | 0.793 | 0.897 | 0.876 | 0.857 |

(GT counts per bin: car 5k–13k; truck 0.6k–2.3k; bus 0.1k–0.6k; pedestrian 40–60 m has only 5 GTs and was excluded.)

- **Degradation grows monotonically with distance**: for car, k8 ≈ clean below 30 m, but at 30–60 m k8 clearly drops and is **below LiDAR-only** — this is spatially where the backfire lives.
- Mean localization error of matched boxes confirms it: car 40–60 m center error clean 0.44 m → k8 0.49 m (LiDAR-only 0.47 m) — misalignment drags localization to (slightly worse than) LiDAR-only levels.
- Mechanism: LiDAR returns get sparse with distance, so BEVFusion relies more on the camera branch for semantic/density compensation exactly at long range. Temporal misalignment projects stale camera features onto moved LiDAR geometry (the ego vehicle keeps driving), and **the corruption lands precisely in the range band where the model depends on the camera most**.

## 4. Mechanism summary — where is the backfire point?

1. **Class axis**: backfire concentrates on small/dynamic, camera-reliant classes (cone, motorcycle, pedestrian, bicycle, car, barrier); LiDAR-dominant large vehicles (truck/bus/trailer/construction vehicle) are safe throughout.
2. **Time axis**: class-level backfire starts at ~500 ms (k6), earlier than the overall mAP crossover (~650 ms).
3. **Scene axis**: day and intersections are the hot spots; at night, misalignment cannot be worse than not using the camera at all.
4. **Range axis**: backfire concentrates at 30–60 m — the LiDAR-sparse band where the model leans on the camera most; close range is unaffected.

**One sentence**: temporal misalignment feeds corrupted camera features exactly into the region where the model needs the camera most — small objects at long range — so fusion ends up worse than LiDAR-only precisely there.

## 5. Method & limitations

- All numbers are paired evaluations on the **common subset n=5712** of the 12 prediction files (fusion clean, k1–k10, LiDAR-only) using the official nuScenes `DetectionEval` (detection_cvpr_2019), identical to the final T1 protocol.
- Per-class AP averages the 4 distance thresholds (0.5/1/2/4 m). Range-bin recall@2m uses custom center-distance matching (<2 m), a different metric from official mAP, used only for spatial localization.
- Scene groups are derived from scene-description keywords (`night` / `intersection`); day = non-night.
- Limitations: per-class numbers are not computed for k1–k5 (can be added on the remote box in ~1.5 min per evaluation if needed); the night group is relatively small (n=566).
