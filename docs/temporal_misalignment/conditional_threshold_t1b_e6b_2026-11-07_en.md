# What still degrades fusion after ego compensation: conditional analysis of T1B (camera) and E6B (LiDAR + camera)

**Date:** 2026-11-07 · **Branch:** `nuo-ft-conditional-threshold` (T1B and E6B code in `scripts/temporal/conditional_threshold_t1b/` and `.../conditional_threshold_e6b/`) ·


## 1. Summary

The question: once ego geometry is corrected, which conditions still make fusion lose its value, and is object-level temporal prediction justified? The suspects were object motion, relative displacement and visibility.

1. **Camera side (T1B).** Calibrating each stale frame with its own capture-time pose removes about 77% of the T1 miss-rate degradation at matched delay (k8). At about 1 s (k12) fusion still beats LiDAR-only overall (gap −1.16 pp) and no class crosses (closest: car −0.14, pedestrian −0.07 pp).
2. **What remains on the camera side is concentrated in fast-moving objects.** Boxes with an object-only displacement of at least 3 object widths per second (8% of boxes) are worse than LiDAR-only at k12 (gap +4.2 pp; car +5.3 pp) and the calibration recovers almost nothing there (13%).
3. **LiDAR side (E6B) is the larger lever.** With both branches compensated, 100 ms more LiDAR delay costs 2.35 pp of miss rate, 505 ms more camera delay 0.90 pp. At 151 ms of stale LiDAR fusion is worse than LiDAR-only overall (+1.25 pp; car +2.6, pedestrian +2.8).
4. **On the LiDAR side the whole aggregate gap comes from boxes that move**, and half of it from the fastest 3% (displacement of at least 0.75 widths per 151 ms; gap +19.6 pp overall, about +20 pp for cars at every distance and visibility). Static boxes contribute nothing in aggregate.
5. **Verdict on the three suspects.** *Object motion*: supported on both sides, the strongest factor. *Relative displacement*: partly supported; r500 keeps an association with the gap even for static objects (an ego-related residual), small on the camera side and larger on the LiDAR side. *Visibility*: not a driver for the stale camera; an independent driver for the stale LiDAR (static cars with 0–40% visibility lose about 6 pp).
6. **Implication.** Object-level temporal prediction is justified for fast-moving objects, mainly on the LiDAR side, but it can at best bring the 151 ms stale-LiDAR fusion to LiDAR-only parity. Occluded and ego-motion-related residuals need other handling.

All results are correlational and use annotation velocity for object motion (an upper bound for what a deployed system can estimate).

## 2. Setup and definitions

- **Metric:** GT miss rate at score >= 0.1; every unmatched GT box is a miss; greedy matching by descending score within (sample, class), 2 m ground-plane center distance. mAP cannot be split per box, so miss rate is the per-box metric.
- **B, D, gap (pp):** B = miss(LiDAR-only) − miss(clean) is the initial benefit; D(cell) = miss(cell) − miss(clean); gap = D − B. gap > 0 means fusion is worse than the LiDAR-only fallback. Gaps are additive over boxes, so the aggregate gap equals the sum of share-of-boxes x group gap over any partition.
- **Uncertainty:** 95% scene-level bootstrap. A cell is `SUPPORTED` with at least 30 trajectories, 10 scenes and a top-1 scene share of at most 0.30. The `RESIDUAL_BACKFIRE` / `BACKFIRE` flag means the lower bound of the gap CI is above 0; it is permissive for large cells, so magnitudes matter more than flags.
- **Conditions:** distance, relative lateral and radial velocity, object speed, ego speed, ego yaw rate, LiDAR points, r500, visibility, night, intersection, and two object-only displacements: `object_disp_1s` = |object velocity| x 1 s / width (camera side) and `object_disp_lidar` = |object velocity| x 0.151 s / width (LiDAR side). r500 = |relative lateral velocity| x 0.5 s / width; it contains the ego translation and not the ego rotation.
- **Adjusted model:** per-box outcome (miss at the probe cell − miss of LiDAR-only) regressed on all condition bins at once (linear probability model, scene-clustered SE, reference = first bin of each condition). A coefficient is the change in the gap in pp with the other conditions held fixed.
- **Populations.** T1B: samples present in all five T1B files, 114,232 GT boxes (5,555 samples with GT; the team's 5,623 includes samples without GT). The k12 file lacks the first key frame of 88–89 scenes (1,795 boxes); counting those boxes as misses, as the first version of the team's FN summary did, created an apparent crossover at k12 that does not exist (corrected in their v2). E6B: all 5,643 samples, 116,027 GT boxes. The two populations differ, so absolute numbers are not comparable across sections.
- **Delays (measured, mean CAM_FRONT offset).** T1B: k2 167, k5 420, k8 672, k10 834, k12 1010 ms. T1: k2 168, k5 423, k8 677 ms; T1 k9/k10 (851/958 ms) are reconstructed. T1 and T1B share the frame index k but not the delay at k10, so the T1 vs T1B comparison uses k8. E6B: camera 167 / 420 / 672 ms, LiDAR about 50 / 151 / 301 ms.

## 3. Camera side: T1B

### 3.1 Aggregate (114,232 boxes)

| setting | miss rate | D (pp) | gap vs LiDAR-only (pp) |
|---|---|---|---|
| fusion clean | 12.87% | 0 | −2.67 |
| LiDAR-only | 15.53% | +2.67 | 0 |
| T1 k2 / k5 / k8 | 14.32 / 16.32 / 17.52% | 1.45 / 3.45 / 4.66 | −1.22 / +0.78 / +1.99 |
| T1B k2 / k5 / k8 | 13.01 / 13.47 / 13.91% | 0.14 / 0.61 / 1.05 | −2.53 / −2.06 / −1.62 |
| T1B k10 / k12 | 14.14 / 14.37% | 1.28 / 1.50 | −1.39 / −1.16 |

At k8 the calibration removes 77% of the T1 degradation (1 − 1.05 / 4.66). T1B stays below LiDAR-only up to k12 (gap −1.16 pp, scene-bootstrap lower bound −1.78).

### 3.2 Per class (pp)

| class | boxes | B | D T1 k8 | D T1B k8 | recovered | gap T1B k12 |
|---|---|---|---|---|---|---|
| car | 49,825 | 2.23 | 4.44 | 1.48 | 0.67 | −0.14 |
| pedestrian | 22,026 | 1.10 | 3.73 | 0.79 | 0.79 | −0.07 |
| trailer | 2,200 | 0.86 | 1.45 | 0.32 | 0.78 | −0.32 |
| barrier | 14,877 | 3.19 | 1.84 | 0.19 | 0.90 | −2.84 |
| traffic cone | 9,822 | 4.32 | 12.58 | 1.25 | 0.90 | −2.31 |
| bus | 1,917 | 3.86 | 3.39 | 0.83 | 0.75 | −2.40 |
| truck | 8,669 | 5.41 | 3.53 | 0.66 | 0.81 | −4.20 |
| motorcycle | 1,746 | 5.21 | 9.05 | 1.95 | 0.78 | −3.04 |
| bicycle | 1,722 | 6.04 | 8.65 | 1.57 | 0.82 | −4.70 |
| construction vehicle | 1,428 | 2.80 | 4.83 | −0.42 | 1.09 | −3.15 |

No class has a gap whose CI lower bound is above 0. Car and pedestrian are closest to the crossing; their camera benefit is nearly used up at about 1 s.

### 3.3 Where the residual is (k12)

Cells with a gap CI above 0 (16 of 143 supported cells; the larger ones):

| group | condition | boxes | B | D | gap [95% CI] | recovered |
|---|---|---|---|---|---|---|
| ALL | object displacement >= 3 widths/s | 9,192 | 1.81 | 5.98 | +4.18 [3.23, 5.26] | 0.13 |
| car | object displacement >= 3 | 7,397 | 1.41 | 6.68 | +5.27 [4.07, 6.56] | 0.07 |
| car | object displacement 1–3 | 4,169 | 1.39 | 3.69 | +2.30 [1.35, 3.13] | 0.23 |
| car | moving | 17,714 | 1.78 | 4.29 | +2.51 [1.83, 3.14] | 0.40 |
| car | approaching | 15,853 | 2.35 | 3.97 | +1.62 [0.99, 2.28] | 0.53 |
| car | r500 >= 1 | 16,502 | 1.45 | 2.74 | +1.28 [0.67, 1.90] | — |
| pedestrian | moving | 15,271 | 0.28 | 0.90 | +0.62 [0.31, 0.95] | — |

Spread of the residual D across the bins of each condition (ALL): object displacement 5.20, distance 3.54, LiDAR points 3.06, approaching 2.70, object speed 1.78, r500 1.25, visibility 1.23, lateral velocity 1.14, ego speed 0.77, yaw rate 0.50, intersection 0.47, night 0.10. Distance and LiDAR points have a wide D spread because B is large for far and sparse boxes, so their gap is below zero; the spread of D alone is not evidence of backfire.

Car, D at k12 by object speed and distance: moving 0.9 near / 4.2 mid / 10.4 far; slow 0.1 / 0.7 / 1.9; static 0.2 / 0.8 / 3.0.

### 3.4 Adjusted effects on the gap (k12, reference = first bin)

| car | effect pp [95% CI] |
|---|---|
| object displacement >= 3 | +7.04 [5.41, 8.67] |
| object displacement 1–3 | +4.09 [2.12, 6.06] |
| r500 >= 1 / 0.5–1 / 0.1–0.5 | +3.36 / +3.42 / +1.47 |
| LiDAR points 3–10 | +1.95 [0.26, 3.64] |
| radial: receding / parallel (vs approaching) | −2.21 / −1.32 |
| far (>39 m) vs near | −1.60 (B effect) |
| visibility | no bin significant |

For ALL the picture is the same (displacement >= 3 +5.07; r500 about +3 to +4; visibility 60–100% about −1.4 relative to 0–40%).

### 3.5 Does r500 carry anything beyond object motion?

Car, gap at k12 by displacement (rows) and r500 (columns):

| displacement | r500 < 0.1 | 0.1–0.5 | 0.5–1 | >= 1 |
|---|---|---|---|---|
| < 0.1 (static) | −4.14 | −2.03 | −0.45 | −0.14 |
| 1–3 | +0.97 | −0.16 | +7.08 | +2.99 |
| >= 3 | +2.96 | +4.06 | +6.96 | +7.11 |

The 1–3 row is noisy and not monotone. Among static cars the gap rises by about 4 pp with r500 but never becomes positive. For a static object r500 is the ego-relative lateral motion, so a small ego-related residual remains after the calibration fix; one plausible cause is the change of viewpoint (untested). Among fast cars r500 amplifies the loss further. Object displacement stays the dominant factor.

## 4. LiDAR side: E6B

### 4.1 Aggregate grid (116,027 boxes; clean 12.83%, LiDAR-only 15.53%, B = 2.69 pp)

| LiDAR delay \ camera delay | 167 ms | 420 ms | 672 ms |
|---|---|---|---|
| about 50 ms (l1) | 14.42% (−1.10) | 14.83% (−0.69) | 15.32% (−0.21) |
| about 151 ms (l3) | 16.78% (+1.25) | 17.17% (+1.64) | 17.69% (+2.16) |
| about 301 ms (l6) | 24.69% (+9.17) | 25.20% (+9.68) | 25.77% (+10.24) |

(gap to LiDAR-only in parentheses). These match the team's E6B-FN note. LiDAR increment c2_l1 → c2_l3 is +2.35 pp (car 2.98, pedestrian 2.11); camera increment c2_l1 → c8_l1 is +0.90 pp (car 1.32, pedestrian 0.50).

### 4.2 Classes at the two probes (gap, 95% lower bound)

| class | camera probe c8_l1 | LiDAR probe c2_l3 |
|---|---|---|
| ALL | −0.21 | +1.25 (0.54) |
| car | +0.95 | +2.61 (1.78) |
| pedestrian | +1.16 | +2.76 (1.74) |
| motorcycle | −0.45 | +3.28 (CI includes 0) |
| trailer | +0.09 | +0.63 (CI includes 0) |
| truck / bus / barrier / construction vehicle | −3.8 / −2.9 / −2.6 / −3.6 | −2.9 / −1.4 / −1.8 / −1.8 |

The camera probe still has 50 ms of LiDAR delay; it is not a camera-only cell.

### 4.3 The fastest boxes carry half of the aggregate gap

Contribution to the aggregate gap at the LiDAR probe (+1.25 pp), by displacement over the LiDAR delay:

| displacement (widths per 151 ms) | share of boxes | gap (pp) | contribution (pp) |
|---|---|---|---|
| < 0.05 (static) | 70.6% | −0.02 | −0.02 |
| 0.05–0.25 | 9.1% | +1.81 | +0.17 |
| 0.25–0.75 | 16.9% | +2.80 | +0.47 |
| >= 0.75 (car above about 34 km/h) | 3.2% | +19.56 | +0.62 |

For the fastest group D = 21.2 pp (B = 1.7); for cars D = 21.7 pp, gap +20.2 pp [lower bound 14.7]. Car D by displacement and distance at the LiDAR probe:

| displacement | far >39 m | mid 21–39 m | near <21 m |
|---|---|---|---|
| < 0.05 | 8.30 | 4.58 | 0.75 |
| 0.05–0.25 | 9.09 | 3.67 | 0.53 |
| 0.25–0.75 | 8.32 | 3.43 | 0.44 |
| >= 0.75 | 25.26 | 20.22 | 20.66 |

Slow boxes show the usual distance gradient; the fastest boxes lose 20–25 pp at every distance. By object speed the contributions are static −0.26, slow +0.17, moving +1.34 (aggregate +1.25).

### 4.4 Two further factors, independent of object motion

**Occlusion.** Car, gap at the LiDAR probe by displacement (rows) and visibility (columns):

| displacement | v1 0–40% | v2 40–60% | v3 60–80% | v4 80–100% |
|---|---|---|---|---|
| < 0.05 | +5.79 | +1.90 | +0.76 | −0.50 |
| 0.25–0.75 | +6.91 | +4.01 | +3.11 | +0.92 |
| >= 0.75 | +19.53 | +21.05 | +21.62 | +20.02 |

Even static cars lose about 6 pp when heavily occluded (8,547 boxes); the fastest cars lose about 20 pp regardless of visibility. Visibility 0–40% accounts for +1.01 pp of the aggregate gap (19% of boxes), LiDAR points 0–2 for +0.73 pp; these overlap with each other and with the motion groups.

**Ego-related residual (r500).** Car, gap at the LiDAR probe by displacement and r500:

| displacement | r500 < 0.1 | 0.1–0.5 | 0.5–1 | >= 1 |
|---|---|---|---|---|
| < 0.05 (static) | −1.97 | +1.07 | +1.93 | +3.95 |
| 0.05–0.25 | +0.80 | −0.34 | +0.19 | +2.33 |
| 0.25–0.75 | +2.26 | +1.39 | +1.18 | +2.21 |
| >= 0.75 | +9.96 | +17.76 | +20.43 | +26.91 |

For static cars the gap rises from −2.0 to +3.9 with r500 (7,000–12,600 boxes per cell), and for the fastest cars from +10 to +27, so r500 and object motion compound.

### 4.5 Adjusted effects at the LiDAR probe (reference = first bin)

| | ALL | car |
|---|---|---|
| displacement >= 0.75 | +20.18 [15.26, 25.10] | +19.59 [14.24, 24.95] |
| displacement 0.25–0.75 | +2.67 | +1.72 |
| r500 >= 1 / 0.5–1 / 0.1–0.5 | +5.18 / +4.24 / +3.36 | +4.82 / +2.49 / +1.72 |
| visibility 80–100% (vs 0–40%) | −5.46 | −5.58 |
| far (>39 m) vs near | n.s. | −2.02 (B effect) |

## 5. The three suspects

| suspect | T1B (stale camera, about 1 s) | E6B (stale LiDAR, 151 ms) |
|---|---|---|
| Object motion | **Supported.** Strongest factor; fast objects still worse than LiDAR-only (gap +4.2 pp; car +5.3), calibration recovers 13% there | **Supported.** Strongest factor; gap about +20 pp at any distance or visibility; the moving boxes explain the whole aggregate gap |
| Relative displacement (r500) | **Weakly supported.** Static cars rise from −4.1 to −0.1 pp with r500; no backfire from this alone | **Supported.** Independent of object motion; static cars −2.0 → +3.9, fast cars +10 → +27 |
| Visibility | **Not supported.** No bin significant for car; about 1.4 pp for ALL | **Supported.** Static cars +5.8 pp at 0–40% vs −0.5 pp at 80–100% |

Distance and LiDAR points are not drivers once B is accounted for: their large D reflects a large initial benefit, and their gap is zero or negative.

## 6. Implications for object-level temporal prediction

- **Camera side:** the aggregate payoff is small. The fast 8% of boxes contribute about 0.3–0.4 pp of miss rate at about 1 s, and fusion stays above LiDAR-only overall. A cheaper alternative is to discount the camera for fast boxes.
- **LiDAR side:** the case is much stronger. Removing the excess loss of all boxes with displacement of at least 0.05 widths per 151 ms would bring the aggregate gap from +1.25 to about 0, i.e. to LiDAR-only parity, not above it. Half of that comes from 3% of boxes.
- **What it will not fix:** heavily occluded boxes and boxes with large ego-relative motion lose value even when static. These overlap with the motion groups, so the shares above are upper bounds for the motion fix.
- **Per delay:** 100 ms of LiDAR delay costs about 2.6 times what 505 ms of camera delay costs (2.35 vs 0.90 pp), so a motion-aware correction has more to gain on the LiDAR branch.

## 7. Limitations

- Correlational; conditions are correlated with each other and the adjusted model is linear and additive (no interactions). Contributions of different conditions cannot be added up.
- Object motion uses annotation velocity (an oracle-style upper bound); r500 contains ego translation but not rotation and is not ego-pose compensation.
- Many cells are tested; single cells are leads. Rare classes are too sparse for condition-level claims.
- Miss rate only; false positives are not analysed (the l6 row adds many FPs, which is why it is not used for conditions).
- T1 k9/k10 delays are reconstructed; T1 and T1B are compared at k8 only. T1B and E6B use different populations.
- E6B mechanism is untested: the E2 report says the sweep list is truncated to `sweeps[k:]`, which would reduce accumulated sweeps at larger k and mix staleness with lower point density. If E5/E6B do the same, part of the occlusion and sparsity effect may be a density effect. To confirm with the team.
- The camera probe of E6B carries 50 ms of LiDAR delay.

## Appendix: notes on team documents (as of 2026-11-07)

- `t1b_fn_analysis_en.md` v2 is consistent with these numbers. Its extrapolation of "about −0.07 pp per 100 ms for car" does not match its own table (car G goes from −0.8 at k8 to −0.14 at k12, about 0.2 pp per 100 ms); the whole-image crossing at 1.5–2 s is consistent with the ALL slope, but car and pedestrian would cross earlier by the same linear extrapolation.
- `e6b_fn_analysis_en.md`: the sentence contrasting E6B with a T1B "mAP safe / miss-rate backfired" divergence is stale (the divergence was retracted); the claim that pedestrian is the only class backfiring on the l1 row at c5/c8 conflicts with its own table (car is +0.3 / +1.0, trailer +0.1 at c8).
