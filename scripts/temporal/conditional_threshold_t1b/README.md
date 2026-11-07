# Conditional analysis of T1B (ego-consistent delayed camera)

Continues the T1 conditional-threshold pipeline in `../conditional_threshold`. Same methodology (per-GT-box miss rate, scene-level bootstrap, audit support rule), applied to **T1B**: each historical camera frame is delivered with its own capture-time calibration, LiDAR current.

**Goal:** once ego geometry is corrected, which conditions still make fusion lose its value, and does the answer justify object-level temporal prediction?

## Questions
1. Does T1B still show conditional backfire (fusion worse than LiDAR-only) for particular object groups, although aggregate mAP stays above LiDAR-only?
2. Which conditions explain the residual degradation after ego geometry is corrected? The suspects: object motion, relative displacement, visibility.
3. How do the conditional patterns differ between T1 and T1B?

## Method
- **Metric:** GT miss rate at score >= 0.1; every unmatched GT box is a miss (corrected E4 definition); matching as in `../conditional_threshold/step2_match.py`.
- **Population:** GT boxes of samples present in every T1B file (a sample absent from a file would otherwise count all its boxes as misses). The k12 file lacks 88 first-key-frame samples (1,795 boxes); the team's FN summary counted them as misses, which created an apparent crossover at k12 that does not exist on the common population.
- **Benefit and degradation:** B = miss(LiDAR-only) − miss(clean); D(k) = miss(T1B, k) − miss(clean); gap(k) = D(k) − B (gap > 0: fusion worse than LiDAR-only). A cell is flagged `RESIDUAL_BACKFIRE` when the 95% scene-bootstrap CI of gap(k12) is above 0.
- **Recovered share:** 1 − D(T1B) / D(T1) at k8, the share of the T1 decay removed by ego-consistent calibration (reported when D(T1) > 1 pp). k8 is used because the mean delay is similar in both protocols (T1 677 ms, T1B about 700 ms); at k10 it is not (T1 about 958 ms reconstructed, T1B 834 ms measured), because the two substitution tables are built differently. Measured T1B delays: k10 about 834 ms, k12 about 1010 ms.
- **1-D screening** for ALL, car, pedestrian, traffic_cone: distance, lateral and radial relative velocity, object speed, ego speed, ego yaw rate, LiDAR points, r500, visibility, night, intersection, and `object_disp_1s` = |object velocity| × 1 s / width (object-only displacement; annotation velocity, so an upper bound for what a real system can estimate).
- **2-D controls:** object speed × distance, object displacement × distance, visibility × distance, LiDAR points × object speed, object displacement × r500.
- **Adjusted model (7e):** per-box outcome miss(T1B, k12) − miss(LiDAR-only) regressed on all condition bins at once (linear probability model, scene-clustered SE, reference = first bin of each condition), for ALL and car. A coefficient is the change in gap (pp) with the other conditions held fixed. A second outcome, miss(T1B) − miss(clean), is the residual degradation. Note that conditions which drive B (distance, LiDAR points) change sign in the gap outcome; read them with the degradation outcome.

## Files
| File | Purpose |
|---|---|
| `run_t1b.py` | The whole T1B analysis: labels, population, class level, 1-D screening, 2-D controls, adjusted model, figures |
| `t1b_config.py` | T1B settings (k = 2, 5, 8, 10, 12), file names, release URL |
| `results/tables/t1b_*.csv`, `results/figures/t1b_*.png`, `results/t1b_key_numbers.json` | Outputs (copied from `ct_results/`) |

## Data
Prediction files from release `predictions-t1b`: `t1b_k{2,5,8,10,12}_cumulative.json.gz`, in the same `team_data/` folder as the T1 files. Missing files are skipped. Nothing here uses the team's condition table: features come from the nuScenes metadata via `../conditional_threshold/step1_features.py`.

## Running on Google Colab
Run the T1 pipeline first (it creates `ct_results/gt_features.csv.gz`, `ego_samples.csv` and `labels/`), then:

```python
%cd /content/drive/MyDrive/nuscenes/script/conditional_threshold_t1b
!python run_t1b.py --root /content/drive/MyDrive/nuscenes
```
Keep `conditional_threshold/` and `conditional_threshold_t1b/` side by side. The run takes about 30 s once the labels exist (a few minutes for the first run).

## Outputs
| Table | Content |
|---|---|
| `t1b_aggregate_miss.csv` | miss rate and FN per setting on the population |
| `t1b_class_level.csv` | B, D, gap, recovered share per class |
| `t1b_screening_1d.csv` | all 1-D cells with CIs and flags |
| `t1b_condition_spread_all.csv` | spread of residual D across the bins of each condition (ALL) |
| `t1b_controls_2d.csv` | 2-D cells |
| `t1b_adjusted_effects.csv` | adjusted effects on the gap and on the degradation (ALL, car) |

| Figure | Content |
|---|---|
| `t1b_aggregate.png` | miss rate vs k, T1 and T1B |
| `t1b_car_degradation.png` | car D by object speed, distance, object displacement; T1 solid, T1B dashed, dotted = B |
| `t1b_recovery_by_condition.png` | D before and after ego-consistent calibration (k8) |
| `t1b_residual_factors_ALL.png`, `t1b_residual_factors_car.png` | adjusted effect of each condition bin on the gap at k12 |

## Results (population: 5,555 samples, 114,232 GT boxes)
- **Aggregate:** miss rate clean 12.87%, LiDAR-only 15.53% (B = 2.67 pp). T1 degradation D at k2 / k5 / k8 = 1.45 / 3.45 / 4.66 pp; T1B D at k2 / k5 / k8 / k10 / k12 = 0.14 / 0.61 / 1.05 / 1.28 / 1.50 pp. T1B at k12 stays below LiDAR-only (gap −1.16 pp).
- **Recovered share at k8:** 77% overall (car 67%, pedestrian 79%, truck 81%, barrier 90%).
- **No class backfires at k12:** every class gap is below 0 and no class-level CI lower bound exceeds 0 (car −0.14, pedestrian −0.07 pp, the closest).
- **Where backfire remains (16 flagged cells, mostly moving-object conditions; car cells for ego speed, yaw rate and intersection are weaker and probably confounded with motion):** all classes with object displacement >= 3 widths per second (gap +4.18 [3.23, 5.26], recovered about 13%); car with displacement >= 3 (+5.27), car moving (+2.51), car approaching (+1.62), car r500 >= 1 (+1.28), pedestrian moving (+0.62).
- **Adjusted model (car, k12):** object displacement is the strongest factor (+7.0 pp for >= 3, +4.1 for 1–3, vs < 0.1). r500 is also positive (+3.4 for >= 0.5) but see the next point. Visibility is not significant for car and small for ALL (about 1.4 pp lower gap for the better-seen bins vs the 0–40% bin).
- **r500 is not an independent factor:** among nearly static cars (object displacement < 0.1) the gap shows no upward trend with r500 (between −4.1 and −0.1 in every r500 bin); among fast cars (displacement >= 3) it is positive and rises (+3.0 for r500 < 0.1, +4.1, +7.0, +7.1 for the higher bins). So r500 matters only where the object itself moves; it behaves as a measure of lateral object motion, not as a separate ego-related residual.
- **Distance and LiDAR points:** residual D is large for far and sparse boxes (car far D = 5.5 pp), but B is large there too, so their gap is below 0; they are not backfire drivers after ego correction.
- **Car, object speed × distance, D at k12:** moving far 10.4, moving mid 4.2, moving near 0.9; static and slow stay at or below 3.0 at any distance.

## Limitations
- T1 and T1B share the frame index k but not the delay beyond k8; use per-protocol measured delays. Figures use k, not ms.
- `object_disp_1s` uses annotation velocity, an oracle-style upper bound; real systems estimate it with error.
- r500 includes ego translation and ignores rotation; it is kept for comparison with T1.
- The adjusted model is linear and additive (no interactions), and the four `missing` dummies (276 boxes without velocity annotation) are identical; ignore them.
- Many cells are tested; single cells are leads, not findings. Rare classes are too sparse for condition-level claims.
- Miss rate, not mAP or FP; the analysis is correlational.
