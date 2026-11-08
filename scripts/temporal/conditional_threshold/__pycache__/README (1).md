# Conditional Backfire Thresholds (T1, stale camera)

This pipeline asks **which objects backfire first** when BEVFusion receives a stale camera image together with current LiDAR. It starts from the team's T1 prediction files and ends with tables and figures. It needs CPU only and does no re-inference.

## Question

In T1 the camera image is replaced by a frame k steps back (k = 1–10, about 77–958 ms) while LiDAR, calibration, pose and GT stay current. On aggregate mAP, fusion falls below LiDAR-only at about 570 ms. This pipeline measures how the onset of that backfire changes with per-object conditions:

- distance and number of LiDAR points
- relative lateral and radial motion, and r
- object speed, ego speed and ego yaw rate
- visibility, night, intersection

## Key findings

1. **Relative lateral displacement r sets both the onset and the size of the backfire.** Pooled over all classes, fusion's miss rate rises above LiDAR-only at **135 ms** for r ≥ 1, 258 ms for 0.5 ≤ r < 1 and 519 ms for 0.1 ≤ r < 0.5. Objects with r < 0.1 never cross. Benefit lost grows from 0.9 to 7.9 points across the same four bins. Both orderings hold in **100%** of 500 scene-bootstrap replicates.
2. **The r effect is robust to distance, ego speed and the score threshold.**
   - Within each car distance bin, a larger r still crosses earlier and loses more.
   - Ego speed adds no consistent effect once r is fixed.
   - The ordering holds at score thresholds 0.1, 0.3 and 0.5; car with r ≥ 1 crosses at 138 / 137 / 135 ms.
3. **Distance sets how much the camera helps, and therefore how much can be lost.** Far cars with r ≥ 1 lose 21.7 points of miss-rate advantage; near cars lose less than 1 point in any r bin.
4. **The source of the displacement does not matter.** At the same r, static and moving cars behave alike: at r ≥ 1 they cross at 133 vs 135 ms and lose 8.1 vs 8.2 points. A static car's r comes entirely from ego motion. Because T1 does not compensate ego pose for the stale image, **uncompensated ego motion is a major driver of the backfire**. An ego-compensated run is still needed to confirm this.
5. **The backfire is a false-negative effect, not a false-positive one.** Total false positives *decrease* with delay at every threshold (−11% to −17% from clean to k10), and at k10 they stay below the LiDAR-only level. Only truck shows a noticeable FP increase (+10%), and it stays below LiDAR-only.
6. **Class level (score ≥ 0.1).**
   - car, pedestrian, traffic_cone, motorcycle, bicycle and construction_vehicle cross LiDAR-only on miss rate.
   - truck, bus and barrier never do.
   - Pooled over all classes the crossover is 305 ms [231, 413], earlier than the ~570 ms mAP crossover.

## Method in brief

| Item | Definition |
|---|---|
| Data | 5,712-sample common subset, 116,027 GT boxes (`gt_common`), 12 settings: fusion clean, k1–k10, LiDAR-only |
| Matching | Same as E4: score ≥ threshold, greedy in descending score order within (sample, class), 2 m ground-plane center distance |
| Miss | **Every unmatched GT box** (corrected E4 definition). The original E4 definition (unmatched *and* the sample has ≥ 1 same-class prediction) is kept for comparison |
| Gap | `miss_rate(fusion, k) − miss_rate(LiDAR-only)` for a group of GT boxes. Below 0 means fusion is better; above 0 means backfire |
| Crossover | Delay after which the gap stays positive (linear interpolation) |
| Benefit lost | `gap(k10) − gap(clean)` in percentage points: how much of the camera's advantage is lost by ~958 ms |
| Uncertainty | 95% CI from a scene-level bootstrap (500 replicates, fixed seed) |
| Support | SUPPORTED needs ≥ 30 trajectories, ≥ 10 scenes and a top-1 scene share ≤ 30% (metadata-audit rule) |
| r500 | `relative lateral speed × 0.5 s / object width`: body-widths the object shifts sideways relative to the ego camera in 0.5 s. Ego motion is included. The 0.5 s factor is a constant, so it changes the bin labels but not how boxes are ranked |

Per-box features follow the metadata audit definitions (`src/metadata_audit`). Bin edges are the audit tertiles from `bin_definitions.json`.

---

## Results

All numbers below are from a full run on the real data with the main miss definition and score ≥ 0.1, unless stated otherwise. Brackets are 95% scene-bootstrap CIs.

### 1. Reconciliation

| Check | Expected | Obtained |
|---|---|---|
| Samples / GT boxes | 5,712 / 116,027 | 5,712 / 116,027 (all 10 per-class counts match) |
| GT boxes without a matching annotation | 0 | 0 |
| Night / intersection samples | 566 / 1,982 | 566 / 1,982 |
| FN, every unmatched box (clean / k10 / LiDAR-only) | 14,892 / 20,903 / 18,015 (corrected E4) | identical |
| FN, original E4 definition | 12,607 / 17,400 / 15,380 | identical |
| FP (clean / k10 / LiDAR-only) | 25,907 / 22,997 / 28,146 (E4) | 25,831 / 22,928 / 28,047 |

FN matches exactly, both in total and per class, so every matching decision agrees with E4. FP is 0.3% lower, mostly for barrier (−37), pedestrian (−15) and car (−9). With identical matches, this means E4 counted slightly more predictions with score ≥ 0.1. The likely cause is that E4 used local copies of the prediction files rather than the release assets. This has not been verified, and it does not affect any miss-rate result.

### 2. Class level

| Class | Miss clean | Miss LiDAR-only | Crossover | Benefit lost (pts) | Crossover, original E4 definition |
|---|---|---|---|---|---|
| car | 0.088 | 0.111 | **279 [217, 367]** | 5.0 | 280 [217, 372] |
| pedestrian | 0.089 | 0.100 | **140 [57, 198]** | 4.1 | 150 [63, 219] |
| traffic_cone | 0.145 | 0.189 | **154 [109, 241]** | 13.3 | 173 [97, 327] |
| motorcycle | 0.230 | 0.283 | 269 [150, 466] | 10.1 | none |
| bicycle | 0.346 | 0.406 | 287 [156, 622] | 10.2 | none |
| construction_vehicle | 0.529 | 0.558 | 308 [125, 644] | 4.7 | none |
| trailer | 0.336 | 0.345 | 523 [0, 910] | 1.5 | 306 [0, 933] |
| truck | 0.225 | 0.279 | none | 4.2 | none |
| bus | 0.143 | 0.182 | none | 3.5 | none |
| barrier | 0.145 | 0.178 | none | 2.2 | none |
| **all classes** | 0.128 | 0.155 | **305 [231, 413]** | 5.2 | 350 [256, 494] |

- car, pedestrian and traffic_cone do not depend on the miss definition.
- motorcycle, bicycle and construction_vehicle backfire only when every unmatched box is counted, i.e. under the corrected E4 definition.
- truck and bus lose some camera benefit but never fall below LiDAR-only, because LiDAR-only misses even more of them.

### 3. Dose–response in r

| Group | r < 0.1 | 0.1 ≤ r < 0.5 | 0.5 ≤ r < 1 | r ≥ 1 |
|---|---|---|---|---|
| **All classes**, crossover | none | 519 [390, 690] | 258 [208, 343] | **135 [117, 156]** |
| All classes, benefit lost | 0.9 [0.6, 1.4] | 3.7 [3.1, 4.4] | 5.2 [4.6, 5.9] | **7.9 [6.9, 8.8]** |
| **car**, crossover | none | 454 [327, 580] | 218 [165, 286] | **138 [120, 158]** |
| car, benefit lost | 1.1 [0.8, 1.7] | 4.7 [3.8, 5.7] | 6.8 [5.9, 7.9] | **7.0 [6.1, 8.0]** |
| **pedestrian**, crossover | none | 176 [0, 716] | 369 [72, 724] | **112 [40, 171]** |
| pedestrian, benefit lost | −0.3 [−1.2, 0.8] | 1.6 [0.7, 2.7] | 2.8 [2.0, 3.7] | **5.2 [4.0, 6.8]** |
| **traffic_cone**, crossover | none (WEAK) | 747 [0, 932] | 340 [0, 554] | **120 [94, 148]** |
| traffic_cone, benefit lost | 0.7 [0.1, 2.6] | 5.8 [2.5, 10.0] | 9.6 [4.3, 16.0] | **16.7 [12.9, 21.1]** |

N per bin: car 12,790 / 10,659 / 10,207 / 16,686; pedestrian 1,477 / 2,403 / 3,321 / 15,154; traffic_cone 1,635 / 433 / 541 / 7,367; all classes 23,370 / 22,482 / 21,383 / 48,510.

**Trend test.** Each cell is the share of 500 scene-bootstrap replicates in which the ordering holds; ≥ 0.95 counts as robust.

| Group | Benefit lost increases across all 4 bins | Benefit lost r ≥ 1 > r < 0.1 | Crossover earlier across all 4 bins | Crossover r ≥ 1 earlier than r < 0.1 |
|---|---|---|---|---|
| all classes | **1.000** | **1.000** | **1.000** | **1.000** |
| car | 0.622 | **1.000** | **0.998** | **1.000** |
| pedestrian | **0.984** | **1.000** | 0.136 | 0.916 |
| traffic_cone | 0.924 | **1.000** | 0.806 | **0.998** |

The car "benefit lost increases across all 4 bins" share is 0.62 only because the top two bins are nearly equal (6.8 vs 7.0). The pedestrian and cone middle bins are noisy, but the r ≥ 1 vs r < 0.1 contrast is robust in every group.

![Car gap curves by condition](figures/car_gap_panels.png)

![Gap curves by r500 for traffic_cone and pedestrian](figures/r500_gap_by_class.png)

### 4. Controls (car)

**Distance × r500.** Crossover in ms; benefit lost in points.

| Distance | r < 0.1 | 0.1 ≤ r < 0.5 | 0.5 ≤ r < 1 | r ≥ 1 |
|---|---|---|---|---|
| far > 39 m | none / 3.4 | 527 [370, 641] / 10.0 | 243 [182, 356] / 16.7 | 180 [155, 204] / **21.7** |
| mid 21–39 m | none / 1.5 | 313 [224, 488] / 4.3 | 152 [88, 263] / 6.3 | 115 [91, 138] / 7.8 |
| near < 21 m | none / 0.0 | none / 0.0 | 434 [202, 793] / 0.7 | 77 [0, 138] / 0.7 |

- Within the far and mid bins, both crossover and benefit lost follow r.
- Across distance bins at a fixed r, far cars lose more, because the camera helps them most at clean.
- Near cars barely benefit from the camera, so their crossovers are not informative.

![Car gap curves by distance](figures/car_gap_by_distance.png)

**r500 × ego speed.** Benefit lost in points.

| r bin | ego slow | ego mid | ego fast |
|---|---|---|---|
| r < 0.1 | 0.5 | 4.2 | 2.0 |
| 0.1 ≤ r < 0.5 | 3.1 | 6.2 | 4.8 |
| 0.5 ≤ r < 1 | 8.1 | 7.2 | 5.9 |
| r ≥ 1 | 8.2 | 8.2 | 6.3 |

Ego speed has no consistent monotone effect once r is fixed. For r ≥ 1 the crossovers are 199 [0, 266] (slow), 149 [117, 179] (mid) and 130 [109, 155] (fast) ms, with overlapping CIs.

**r500 × object speed (static vs moving at the same r).** Crossover in ms; benefit lost in points.

| r bin | static | slow | moving |
|---|---|---|---|
| r < 0.1 | none / 0.3 | none / 2.1 | 851 / 1.5 |
| 0.1 ≤ r < 0.5 | 439 / 5.6 | 482 / 4.6 | 390 / 4.4 |
| 0.5 ≤ r < 1 | 202 / 7.8 | 223 / 4.8 | 234 / 9.0 |
| r ≥ 1 | **133 [89, 184] / 8.1** | 145 [124, 176] / 5.3 | **135 [109, 165] / 8.2** |

At the same r, static and moving cars have overlapping CIs and nearly equal crossovers and losses. What matters is the total relative lateral displacement, whatever its source. The "slow" bin loses somewhat less at r ≥ 1 (5.3 points).

Pedestrians are an exception. At r ≥ 1, static pedestrians lose 12.0 points and moving ones only 2.8. This is not yet explained and is not interpreted here.

### 5. Score-threshold sensitivity

**FN growth from clean to k10:** +40.4% (score ≥ 0.1), +26.1% (≥ 0.3), +15.8% (≥ 0.5).

**Class-level crossovers (ms).**

| Class | ≥ 0.1 | ≥ 0.3 | ≥ 0.5 |
|---|---|---|---|
| car | 279 [217, 367] | 289 [241, 338] | 338 [291, 396] |
| pedestrian | 140 [57, 198] | 177 [133, 243] | 317 [235, 413] |
| traffic_cone | 154 [109, 241] | 148 [108, 209] | 123 [76, 172] |
| motorcycle | 269 [150, 466] | 362 [218, 851] | 324 [167, 603] |
| bicycle | 287 [156, 622] | 484 [185, 928] | 418 [209, 666] |
| truck / bus / barrier | none | none | none |
| all classes | 305 [231, 413] | 328 [245, 447] | 414 [347, 518] |

**All classes, r500 bins.** Crossover in ms / benefit lost in points.

| r bin | ≥ 0.1 | ≥ 0.3 | ≥ 0.5 |
|---|---|---|---|
| r < 0.1 | none / 0.9 | none / 1.2 | none / 1.2 |
| 0.1 ≤ r < 0.5 | 519 / 3.7 | 556 / 5.5 | none / 5.4 |
| 0.5 ≤ r < 1 | 258 / 5.2 | 328 / 7.6 | 403 / 7.9 |
| r ≥ 1 | **135 / 7.9** | **147 / 11.0** | **166 / 11.2** |

- Higher thresholds shift most class-level crossovers later. traffic_cone is the exception.
- The r ordering is unchanged at every threshold.

### 6. False positives

| Score threshold | FP clean | FP k10 | Change | FP LiDAR-only |
|---|---|---|---|---|
| ≥ 0.1 | 25,831 | 22,928 | −11.2% | 28,047 |
| ≥ 0.3 | 6,919 | 5,905 | −14.7% | 6,822 |
| ≥ 0.5 | 2,593 | 2,158 | −16.8% | 2,197 |

**Per class at score ≥ 0.1** (clean → k10, LiDAR-only):

| Class | clean | k10 | LiDAR-only |
|---|---|---|---|
| car | 7,367 | 6,730 | 8,128 |
| pedestrian | 6,579 | 5,617 | 6,890 |
| traffic_cone | 4,069 | 2,888 | 4,272 |
| barrier | 3,832 | 3,686 | 3,959 |
| bicycle | 407 | 293 | 475 |
| motorcycle | 182 | 163 | 274 |
| truck | 2,207 | 2,428 | 2,767 |

- **truck is the only class whose FP rises noticeably (+10%)**, and it stays below LiDAR-only.
- Split by predicted distance, only four bins sit above LiDAR-only at k10, all by small margins: bus mid (+12), construction_vehicle near (+8), bus near (+3) and trailer near (+1). In each of these, FP barely changes from clean to k10.
- There is no sign of misalignment creating phantom detections.

![False positives vs camera delay](figures/fp_vs_delay.png)

### 7. Robust conditional cells

Of 406 class × condition × bin cells, 379 are SUPPORTED. **17 cells pass the robust screen** (§ How to read the outputs):

- **11 car:** r ≥ 1, lateral fast, yaw high, ego mid/fast, receding, approaching, moving, and 0.1 ≤ r < 0.5 (later).
- **1 pedestrian:** visibility 60–80% (later).
- **5 traffic_cone.**

Four of the 17 are flagged `small_clean_benefit`, meaning the camera barely helps there at clean:

| Class | Cell | Why it is flagged |
|---|---|---|
| car | > 10 LiDAR points | gap at clean only −0.1 pts |
| car | 20–30 m | gap at clean only −0.5 pts |
| traffic_cone | 0–10 m | fusion already worse than LiDAR-only at clean |
| traffic_cone | visibility 0–40% | fusion already worse than LiDAR-only at clean |

Their early crossovers do **not** mean faster degradation. Every unflagged early car cell is a proxy for relative lateral motion, consistent with §3.

### What can and cannot be claimed

**Supported:**
- The onset and size of the backfire scale with r for car, traffic_cone, pedestrian and all classes pooled.
- This holds within distance bins, regardless of ego speed and score threshold, and equally for static and moving cars.
- The backfire is driven by false negatives.

**State with care:**
- r mixes ego motion and object motion. The static-vs-moving result suggests uncompensated ego motion is a major contributor, but only an ego-compensated experiment can confirm it.
- These are miss-rate crossovers, not mAP crossovers.

**Not supported:**
- Condition-level conclusions for rare classes (motorcycle, bicycle, construction_vehicle, trailer, bus, truck).
- Causal claims.

---

## Files

| File | What it does | Main outputs |
|---|---|---|
| `common.py` | Constants, paths, I/O, geometry, bootstrap and crossover functions | — |
| `step0_download.py` | Downloads `gt_common` and the 12 prediction files from the team release `predictions-v1` | `<data>/*.json(.gz)` |
| `step1_features.py` | Computes per-GT-box features from the nuScenes metadata; reconciles box and scene counts | `gt_features.csv.gz`, `ego_samples.csv` |
| `step2_match.py` | Labels every GT box as hit or miss and counts false positives, for all 12 settings at score thresholds 0.1 / 0.3 / 0.5; reconciles FN/FP with E4 | `labels/<setting>.npz`, `labels/<setting>_fp.csv` |
| `step3_crossovers.py` | Class-level and conditional crossovers under both miss definitions; robust-finding screen | `tables/class_crossovers.csv`, `tables/conditional_cells_*.csv`, `tables/robust_findings.csv` |
| `step4_controls.py` | r dose–response, controls, r trend bootstrap, threshold sensitivity, false-positive tables | `tables/r500_bins.csv`, `tables/car_distance_x_r500.csv`, `tables/car_r500_x_ego_speed.csv`, `tables/r500_x_object_speed.csv`, `tables/r_trend_bootstrap.csv`, `tables/threshold_sensitivity.csv`, `tables/fp_*.csv`, `key_numbers.json` |
| `step5_plots.py` | Figures | `figures/car_gap_panels.png`, `figures/r500_gap_by_class.png`, `figures/car_gap_by_distance.png`, `figures/fp_vs_delay.png` |
| `run_all.py` | Runs steps 1–5 in order (add `--download` to run step 0 first) | everything above |

The figures shown in this README are copied from `<out>/figures/` into the repository's `figures/` folder.

## Data layout

```
<root>/
  v1.0-trainval/      nuScenes v1.0-trainval metadata JSON files (from nuscenes.org, about 0.4 GB)
  team_data/          gt_common.json, fusion_cumulative.json.gz, lidar_cumulative.json, k1..k10_cumulative.json.gz
  ct_results/         created by the pipeline
```

Override any folder with `--meta`, `--data` or `--out`. Sensor blobs (images and point clouds) are **not** needed.

## Running on Google Colab

1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Put this folder on Drive, for example `MyDrive/nuscenes/conditional_threshold/`.
3. Run the pipeline:
   ```python
   !pip -q install ijson
   %cd /content/drive/MyDrive/nuscenes/conditional_threshold
   !python run_all.py --root /content/drive/MyDrive/nuscenes            # add --download on first use
   ```

A full run takes about 8 minutes on a standard Colab CPU runtime (7.4 min measured):

| Step | Time |
|---|---|
| Step 1 (streams the metadata) | about 2 min |
| Step 2 | about 20 s per prediction file |
| Steps 3–5 | under 1 min |

Steps 1 and 2 skip work that is already on disk, so an interrupted run can simply be restarted.

Any step can also be run on its own:

```bash
python step3_crossovers.py --root /path/to/nuscenes
```

Requirements are in `requirements.txt`: Python ≥ 3.9, numpy, pandas, matplotlib, and optionally ijson. `ijson` streams the large metadata files; without it, `json.load` is used and needs more RAM.

## How to read the outputs

- **`class_crossovers.csv`**: one row per (definition, class) with `miss_clean`, `miss_k10`, `miss_lidar`, `crossover_ms` [`ci_lo`, `ci_hi`], `kind` and `benefit_lost` [`bl_lo`, `bl_hi`]. `kind` takes these values:

  | `kind` | Meaning |
  |---|---|
  | `cross` | crossover found |
  | `always` | fusion already worse at clean |
  | `none` | gap never stays positive up to about 958 ms |
  | `skip` / `empty` | not enough data |

- **`conditional_cells_all.csv`**: one row per (class, condition, bin). `shift_ms` is the crossover minus the class reference.
- **`robust_findings.csv`**: cells that pass all of these under **both** miss definitions:
  - SUPPORTED
  - `|shift| > 50 ms`, with the same sign under both definitions
  - class reference outside the 95% CI
  - CI width ≤ 400 ms

  `small_clean_benefit` marks cells where the camera barely helps at clean (`gap_clean > −1 pt`). An early crossover there does not mean fast degradation.
- **`r_trend_bootstrap.csv`**: share of bootstrap replicates in which the r ordering holds, for benefit lost and for crossover. Values ≥ 0.95 mean robust.
- **`threshold_sensitivity.csv`**: class-level and r-bin results at score ≥ 0.1, 0.3 and 0.5.
- **`fp_by_class_distance.csv`**: false positives by predicted class × predicted distance at clean, k5, k10 and LiDAR-only.

## Limitations

- **Metric.** The analysis uses miss rate, not mAP, because per-box attributes cannot split mAP. Miss rate does not capture localization or score ranking. False positives are reported separately, split by prediction attributes.
- **Crossover vs. benefit size.** Crossover times depend on how large the clean camera benefit was, so always read them together with benefit lost.
- **Ego motion.** r mixes object motion and ego motion, and T1 does not compensate ego pose for the stale image. The ego-motion interpretation needs an ego-compensated sweep to confirm.
- **Delays and r.** The k9/k10 delays are estimates. r500 assumes constant velocity and is used as a ranking variable for longer delays.
- **Multiple comparisons.** Many cells are tested, so single cells are leads only. The primary evidence is the r trend test, the robust-finding screen and the controls.
- **FP reconciliation.** FP totals differ from E4 by 0.3%, probably because of different prediction-file copies (see §1).


