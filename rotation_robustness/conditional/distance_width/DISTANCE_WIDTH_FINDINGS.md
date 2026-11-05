# Distance × clean projected width

**Clean projected width retains some conditional association, but does not provide a consistent smaller-width risk ordering beyond distance. Its strong marginal trend is substantially entangled with distance composition. It cannot be promoted to a Spatial equivalent of Temporal r500.**

This focused extension reuses the accepted 53,043 Car identities in the 6,019-sample cohort, 27 hit conditions and fixed width geometry. There is no new inference, matching, geometry extraction or audit. Definitions are B=LiDAR miss−clean miss, D=noisy miss−clean miss and G=D−B, in percentage points (pp). These are object miss rates, not mAP/NDS. Each noisy σ retains five separate seeds.

## 1. Does smaller width imply larger degradation within distance?

**Not generally.** At σ=5°, mean D is:

| Distance | <50 px | 50–100 px | 100–200 px | ≥200 px |
|---|---:|---:|---:|---:|
| 0–10 m | empty | empty | 0.000* | 0.047 |
| 10–20 m | empty | 0.959* | 0.726 | 0.599 |
| 20–30 m | 0.000* | 2.540 | 2.323 | 3.013 |
| 30–40 m | 10.698 | 10.035 | 10.674 | 8.460 |
| ≥40 m | 13.359 | 19.096 | 19.519 | 11.984 |

*Limited support: N<200 or fewer than 10 contributing scenes. Table values are descriptive five-seed means, not significance declarations.*

At 10–20 m, the two supported groups differ only modestly (0.726 vs 0.599pp). At 20–30 m, the widest group has the largest D, reversing the marginal ordering. At 30–40 m the widest group is less degraded, but intermediate widths do not form a monotonic order. At ≥40 m the intermediate groups degrade most. The <50px group has less degradation than both 50–100px and 100–200px groups. Some residual width association is present, particularly in far groups, but its shape is inconsistent with a universal inverse-width rule.

## 2. Is crossover ordering consistent?

**No.** At 20–30 m, 100–200px crosses in the observed 1°–2° bracket, while both 50–100px and ≥200px cross in 2°–3°. At 30–40 m, the two wider groups cross in 1°–2°, while the two narrower groups cross in 2°–3°. At ≥40 m, intermediate widths cross in 1°–2°, but both extremes cross in 2°–3°.

Initial benefit matters. At 30–40 m, ≥200px crosses earlier than <50px even though D5 is smaller (8.460 vs 10.698pp), partly because B is smaller (2.836 vs 3.876pp). At 10–20 m, 100–200px already has B=−0.133pp: its positive G is not a newly induced loss of clean advantage. The ≥200px group has B=0.379pp and a 3°–4° crossing despite only 0.599pp degradation at 5°. This is a small-benefit crossing. By contrast, ≥40m intermediate-width cells have both substantial B (7.934/9.308pp) and large D5 (19.096/19.519pp).

Crossings are adjacent observed mean-level brackets, with clean σ=0 included when needed. They are not interpolated physical thresholds or statistically certified onsets. Mean±seed-SD curves and the retained paired scene intervals reflect appreciable variability, especially far/small-support groups.

## 3. Does distance remain associated after fixing width?

**Yes, strongly in the descriptive means.** For 100–200px, D5 increases from 0.726 to 2.323 to 10.674 to 19.519pp across 10–20, 20–30, 30–40 and ≥40m. All four cells have substantial support. For ≥200px, it increases from 0.047 to 0.599 to 3.013 to 8.460 to 11.984pp across the five distance groups. The 50–100px supported cells similarly rise from 2.540 to 10.035 to 19.096pp. Fixing coarse width bins does not absorb the distance association.

## 4. Comparison with accepted Distance × r_rot

The [accepted r_rot study](../r_rot/README.md) found only partial within-distance ordering, with large residual distance effects. For example, r_rot D5 at 20–30m was nearly flat (2.570–2.763pp), and at 30–40m was 10.174 vs 10.126pp; at ≥40m, higher r did separate D (15.028 vs 18.989pp). Width is complementary evidence that projected scale alone is insufficient. Its crossover and D ordering are less coherent with a simple geometric-risk direction, even though some far-width contrasts are numerically larger. Different support/composition and bins preclude a calibrated strength ranking. No model comparison or new r_rot calculation is performed.

## 5. Spatial analogue of Temporal r500, or distance proxy?

**Width is a useful descriptive scale feature, strongly mixed with distance; it is not a demonstrated analogue of r500.** [Temporal r500](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/nuo-ft-conditional-threshold/scripts/temporal/conditional_threshold) includes relative lateral motion over a reference delay normalized by object width. Raw clean pixel width contains no motion or perturbation displacement. Spatial r_rot includes a first-order rotation-induced projection sensitivity divided by width and is physically closer to a displacement normalization, although its accepted explanation is also partial.

The conditional patterns reject a universal smaller-width explanation. They also do not justify calling width *only* a distance proxy, since some within-bin differences remain. These coarse strata cannot quantify the fraction explained by distance or isolate a width effect. The strong marginal trend is substantially composition-dependent; width supplies limited, nonmonotonic additional descriptive information.

## Composition and scope

Among 0–10m objects, 92.9% are ≥200px; this drops to 2.9% at ≥40m. Only one <50px object occurs below 30m. There are 14 objects at 0–10m/100–200px and 146 at 10–20m/50–100px. These are flagged, not interpreted as robust contrasts. Far extreme-width cells (258, 792 and 257 objects) pass the predefined support rule but retain substantial seed variation. Of 20 known-width cells, 17 are populated and 14 pass the rule. The 379 missing-width observations (377 near, 2 at 10–20m) are separately retained; all have zero accepted miss-rate differences.

Width uses the accepted largest-clean-projected-width camera selection, full horizontal projected corner span without clipping, and unchanged validity rules. It depends on range, physical size, orientation and camera selection. Neither occlusion nor network cropping is modeled. LiDAR support, continuous within-bin distance/width, camera direction and scene composition remain confounders. GT-derived widths and injection σ are research diagnostics, with no deployment-readiness claim. No causal claim or multiplicity-adjusted ordering test is made.

## Figures and tables

- [A — B/D/G heatmaps with counts](results/figures/A_BDG_heatmaps.png)
- [B — G–σ curves within distance](results/figures/B_conditional_crossover.png) (σ=0 is clean; panels use separate y scales; dotted lines flag limited support)
- [C — D–σ curves within width](results/figures/C_residual_distance.png) (supported cells only)
- [D — composition counts and within-distance proportions](results/figures/D_composition.png) (proportions rounded; see counts for tiny cells)
- [Compact table](results/compact.csv) · [All 27 rates and seed differences](results/all_conditions.csv) · [Full summary and scene intervals](results/summary.json)

All figures also have PDF/SVG exports. Missing-width groups are shown in A/D and tables, not treated as an ordered width category in B/C. [Reproduction](README.md) · [Focused validation](VALIDATION.md).

The five questions are answered. This extension stops here without changing previous results, exploring alternative definitions, adding other pairs or beginning robustness development.
