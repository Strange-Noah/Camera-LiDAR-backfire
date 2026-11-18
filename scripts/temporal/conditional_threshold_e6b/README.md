# E6B conditional analysis

Continues `../conditional_threshold` (T1) and `../conditional_threshold_t1b` (T1B); keep the three folders side by side.

**E6B** = joint misalignment with both branches ego-consistent: the camera frame is the k_c-th past frame with its own capture-time calibration (as T1B), the LiDAR is the k_l-th past sweep with ego-motion compensation (as E5). Grid: camera k_c in {2, 5, 8} (measured mean delay 167 / 420 / 672 ms) x LiDAR k_l in {1, 3, 6} (about 50 / 151 / 301 ms). Cell names `e6b_c{kc}_l{kl}`.

**Goal:** after both branches are compensated, which GT boxes still lose fusion value, and is the loss driven by the LiDAR delay or by the camera delay?

## Method
- **Metric and population:** GT miss rate at score >= 0.1, every unmatched GT box a miss, matching as in `../conditional_threshold/step2_match.py`. Population = samples present in every E6B file (all 5,643 samples, 116,027 GT boxes; the team's E6B-FN note uses the same boxes). False positives are not analysed (the l1 and l3 rows are almost pure FN damage; the l6 row adds many FPs).
- **Definitions (pp):** B = miss(LiDAR-only) − miss(clean); D(cell) = miss(cell) − miss(clean); gap(cell) = D − B (gap > 0: fusion worse than LiDAR-only).
- **Probe cells:** `e6b_c2_l3` (LiDAR 151 ms, camera 167 ms) for the LiDAR side; `e6b_c8_l1` (camera 672 ms, LiDAR 50 ms) for the camera side. The camera probe still carries 50 ms of LiDAR delay, so it is not a camera-only cell; use T1B for that. Increments: LiDAR = D(c2_l3) − D(c2_l1); camera = D(c8_l1) − D(c2_l1).
- **Object displacement over the LiDAR delay:** `object_disp_lidar` = |object velocity| x 0.151 s / width (annotation velocity, an upper bound for what a real system can estimate). `object_disp_1s` (1 s horizon) is kept for comparison with T1B.
- **Analyses:** 8a class level; 8b 1-D screening (ALL, car, pedestrian) with scene-bootstrap CIs; 8c 2-D controls (object speed x distance, displacement x distance, visibility x distance, LiDAR points x object speed, displacement x r500, displacement x visibility); 8e adjusted model (per-box gap regressed on all condition bins at once, scene-clustered SE, reference = first bin of each condition); 8f contribution of each bin to the aggregate gap (gaps are additive over boxes; bins of one condition sum to the aggregate).
- The BACKFIRE flag (95% CI lower bound of the gap above 0) is permissive here (91 of 121 cells flagged at the LiDAR probe); read magnitudes and contributions, not flag counts.

## Files
| File | Purpose |
|---|---|
| `run_e6b.py` | The whole E6B analysis (labels, population, class level, screening, 2-D controls, adjusted model, contributions, figures) |
| `results/tables/e6b_*.csv`, `results/figures/e6b_*.png`, `results/e6b_key_numbers.json` | Outputs (copied from `ct_results/`) |

## Data and running
Prediction files from release `predictions-e6b`: `e6b_c{2,5,8}_l{1,3,6}_cumulative.json.gz`, in `team_data/` next to the T1 files. Run the T1 pipeline first (it creates `gt_features.csv.gz`, `ego_samples.csv`, `labels/`), then:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/nuscenes/script/conditional_threshold_e6b
!python run_e6b.py --root /content/drive/MyDrive/nuscenes
```
First run: about 1–2 minutes per file for the labels; later runs about 1 minute.

## Outputs
Tables: `e6b_aggregate_miss.csv`, `e6b_class_level.csv`, `e6b_screening_1d.csv`, `e6b_condition_spread_all.csv`, `e6b_controls_2d.csv`, `e6b_adjusted_effects.csv`, `e6b_gap_contribution.csv`.
Figures: `e6b_grid_gap.png` (3x3 gap to LiDAR-only for ALL, car, pedestrian), `e6b_car_degradation.png` (car D vs LiDAR delay by object speed, distance, displacement), `e6b_gap_by_condition.png`, `e6b_residual_factors_{lidar,camera}_probe_{ALL,car}.png` (adjusted effects).

## Results (population: 5,643 samples, 116,027 GT boxes)
- **Aggregate** (miss rate, gap to LiDAR-only 15.53%; clean 12.83%, B = 2.69 pp): l1 row 14.42 / 14.83 / 15.32 (gap −1.10 / −0.69 / −0.21); l3 row 16.78 / 17.17 / 17.69 (+1.25 / +1.64 / +2.16); l6 row 24.69 / 25.20 / 25.77 (+9.2 to +10.2). Same numbers as the team's E6B-FN note.
- **LiDAR delay is the larger lever:** from c2_l1 to c2_l3 (+100 ms LiDAR) D rises by 2.35 pp (car 2.98, pedestrian 2.11); from c2_l1 to c8_l1 (+505 ms camera) by 0.90 pp (car 1.32, pedestrian 0.50).
- **Classes at the LiDAR probe (gap, 95% lower bound):** car +2.61 (1.78), pedestrian +2.76 (1.74), ALL +1.25 (0.54); motorcycle +3.28 (CI includes 0); truck, bus, barrier, construction vehicle stay below LiDAR-only.
- **Fast objects dominate:** boxes with displacement >= 0.75 widths per 151 ms (3.2% of boxes; a car above about 34 km/h) have D = 21.2 pp, gap +19.6 pp (95% lower bound 14.3; adjusted effect +20.2 pp); for cars the damage is 20–25 pp at every distance, while slower boxes show the usual distance gradient (0.4 near to 8–9 far).
- **Contribution to the aggregate gap (+1.25 pp):** displacement < 0.05: −0.02; 0.05–0.25: +0.17; 0.25–0.75: +0.47; >= 0.75: +0.62. Static boxes contribute nothing; the boxes with some motion explain the whole aggregate gap, half of it from the fastest 3%.
- **Occlusion is an independent factor:** static cars with visibility 0–40% have gap +5.8 pp vs −0.5 pp at 80–100%; fast cars lose about 20 pp at every visibility level. In the adjusted model (car) visibility 80–100% is −5.6 pp vs 0–40%.
- **r500 is independent of object displacement:** among static cars the gap rises from −2.0 (r500 < 0.1) to +3.9 (r500 >= 1); among fast cars from +10 to +27. For a static object r500 is the ego-relative motion, so an ego-related residual remains after compensation (mechanism untested).
- **Not a clean decomposition:** the groups above overlap (fast objects, r500, visibility, LiDAR points are correlated); contributions of different conditions cannot be added up.

## Limitations
- The camera probe `c8_l1` is not camera-only; the grid has no fresh-LiDAR cell.
- Open question for the team: whether the stale-LiDAR run accumulates fewer sweeps at larger k (the E2 report says the sweep list is truncated to `sweeps[k:]`), which would mix staleness with lower point density and could explain part of the occlusion effect. Unverified.
- `object_disp_lidar` uses annotation velocity (oracle-style); real systems estimate it with error.
- The adjusted model is linear and additive; correlational throughout; miss rate only (no FP); many cells tested, single cells are leads.
