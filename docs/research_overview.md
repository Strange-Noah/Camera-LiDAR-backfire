# Research Overview: When Fusion Backfires

## 1. Research Problem

Camera–LiDAR fusion combines complementary sensing modalities. Its advantage depends on the spatial and temporal consistency of their observations.

We investigate three related questions:

1. Under what spatial or temporal errors does fusion lose its advantage over LiDAR-only detection?

2. How do these failures depend on the objects, geometry and motion present in a scene?

3. To what extent can online calibration and ego-motion consistency restore detection performance?


The project is organized into two parallel research tracks, Spatial and Temporal. Calibration and conditional threshold analysis are extensions within these tracks, not separate projects.

## 2. Experimental Framework

- **Primary detection system:** BEVFusion.

- **Dataset:** nuScenes validation data.

- **Reference systems:** Clean fusion and LiDAR-only detection.

- **Primary detection metrics:** nuScenes mAP and NDS.

- **Additional analysis:** Per-class performance, object-level matching, miss rates, false positives, rotational errors and scene-level uncertainty.

- **Backfire:** The evaluated fusion system performs worse than its matched LiDAR-only reference under a stated metric and evaluation population.


The spatial and temporal teams used different experimental environments and, in several cases, different paired validation subsets. Comparisons must respect those differences.

## 3. Spatial Track

### 3.1 Rotation robustness

The primary spatial experiment applied persistent, independently sampled camera-local rotational errors to six cameras.

Five Gaussian perturbation levels (σ = 1°–5°) were tested with five seeds, using all 6,019 nuScenes validation frames per condition.

The published five-seed means were:

|Rotation σ|Fusion mAP (%)|Fusion NDS (%)|
|---|---|---|
|1°|67.706|70.849|
|2°|66.322|70.087|
|3°|65.279|69.529|
|4°|64.484|69.080|
|5°|63.833|68.751|

In that published experiment, clean fusion achieved 68.5474% mAP / 71.3865% NDS, and LiDAR-only achieved 64.6647% mAP / 69.2510% NDS.

The five-seed means first fell below LiDAR-only at the tested 4° level. However, the reported seed-level uncertainty intervals for the fusion-minus-LiDAR margin included zero at 4° and 5°, so these observations should not be described as an established population-level crossover threshold.

Primary reference:

[Rotation robustness — methods, statistics and plots](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/5e8bf5701ad4db1da13d73eabd361eb10668a10e/rotation_robustness/README.md)

### 3.2 Online calibration recovery

An OCaMo-based online calibration implementation was coupled to BEVFusion. The system maintained independent camera-specific rotational correction tracks and evaluated corrected detection over the same 25 spatial noise conditions.

The completed corrected experiment contains 150,475 prediction frames.

Principal reported findings:

- 76.6% of camera trajectories experienced a reduction in rotational calibration error.

- Mean Corrected minus Perturbed mAP: +0.1845 percentage points.

- Mean Corrected minus Perturbed NDS: +0.1112 percentage points.

- At σ = 4° and 5°, corrected fusion remained below the LiDAR-only reference in the mean.


The geometric improvement rate is a proportion of trajectories whose error decreased, not the percentage of calibration error removed.

**Interpretation:** Online geometric calibration frequently improves estimated alignment, but this particular implementation delivered only limited downstream detection recovery under the tested spatial perturbations.

**Important limitation:** Historical Spatial and newer Corrected predictions failed strict numerical identity checks. Small metric differences cannot be attributed exclusively to OCaMo without qualification. The final results are not evidence that all calibration methods are ineffective.

The principal presentation figure and its provenance are indexed in [curated results](../results/project_summary/README.md). The earlier published OCaMo baseline is available [here](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/85f489a840a65bacbbe139b9781fc062faac58e0/docs/ocamo/FULL_METHOD_RESULTS.md); it is not the final integrated BEVFusion experiment.

## 4. Temporal Track

The temporal experiments address both camera and LiDAR staleness.

The research distinguishes:

- **Naive camera staleness:** Older camera images substituted while current-frame alignment metadata is retained.

- **Ego-consistent camera staleness:** Stale images delivered with their own capture-time geometric information.

- **Naive LiDAR staleness:** Older LiDAR observations substituted without appropriate motion compensation.

- **Ego-motion-compensated LiDAR:** Stale LiDAR geometry transformed consistently with ego motion.

- **Joint misalignment:** Both sensor branches delayed, with comparisons between naive and consistent representations.


### 4.1 Main findings

In the later paired experiments documented by the temporal team:

|Experiment|Main observation|
|---|---|
|Naive camera staleness|Aggregate mAP backfire at approximately 700–800 ms|
|Ego-consistent camera staleness|No aggregate mAP crossover through approximately 1.0 s of measured delay|
|Naive LiDAR staleness|Backfire at the first tested approximately 50 ms delay|
|Ego-motion-compensated LiDAR staleness|Crossover between approximately 50 and 100 ms|
|Joint misalignment|Consistent alignment mitigates degradation, but stale dynamic LiDAR content remains a major limitation|

These are experiment-specific findings, with evaluation subsets varying between approximately 5,623 and 5,856 frames in the principal synthesis.

An earlier T1/E1 study estimated a camera backfire crossover near 570 ms on a different paired protocol and population. It is retained as a historical result, not silently substituted for the later synthesis.

Source:

[Complete Temporal Synthesis — T1/T1B and E-series](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/2e8dd75aecf6be6f2de8249f5cfce02c35fb8167/results/temporal_misalignment/temporal_misalignment_synthesis_en.md)

### 4.2 Conditional backfire

The conditional analysis evaluates the same T1 stale-camera family at the object level rather than inferring all failures from aggregate detection metrics.

Its normalized lateral displacement variable `r500` measures lateral motion relative to object width over a fixed 0.5-second horizon.

Reported pooled object-level miss-rate crossover estimates include:

|r500 group|Approximate crossover|
|---|---|
|r500 ≥ 1|135 ms|
|0.5 ≤ r500 < 1|258 ms|
|0.1 ≤ r500 < 0.5|519 ms|
|r500 < 0.1|No crossover in the tested range|

This analysis also identifies distance-dependent initial camera benefit, relative motion, and the importance of ego-motion inconsistency.

The threshold values refer to object-level hit/miss comparisons under the stated matching and score protocol; they are not general deployment guarantees.

Source:

[Conditional Threshold Analysis](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/58b49c53e3bf452ad6b87438c5a94b72ad852cac/scripts/temporal/conditional_threshold/README.md)

## 5. Related Selective Calibration Study

A complementary study explored a different calibration assumption: a shared synthetic LiDAR-frame Roll error observed through six cameras.

It constructed six camera-specific kernel-sum score curves, combined their normalized features, trained a Ridge regression model and applied a frozen correction gate.

The final independent 27-scene evaluation reported:

- +0.5° injection: 3.6% correction coverage, accepted MAE 0.036°.

- −0.5° injection: 9.0% coverage, accepted MAE 0.065°.

- Zero clean false-trigger scenes out of 27; one-sided exact 95% upper confidence bound 10.5%.


The low accepted MAE applies only to accepted observations. The model does not establish reliable general three-axis, camera-independent calibration and was not evaluated as a BEVFusion detection-recovery system.

This related work should remain distinct from the independent six-camera spatial perturbation experiment.

Source:

[Selective Calibration — Branch README](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/6253452f0131e19791b1a3deee566ccede2ca1b8/README.md)

## 6. Cross-Track Interpretation

The completed experiments support three conclusions:

1. Fusion can lose its advantage over LiDAR-only detection when sensor alignment deteriorates.

2. Vulnerability is conditional: the amount, type and geometry of misalignment matter, together with object and scene characteristics.

3. Improving alignment does not automatically guarantee recovery of downstream detection performance.


These observations motivate reliability-aware and potentially gated fusion. However, a validated universal switching or fusion-gating policy is **not** among the completed deliverables.

## 7. Limitations and Reproducibility

- A single principal BEVFusion model family was studied.

- Spatial and temporal experiments used different perturbation designs.

- Some temporal experiments use different matched validation subsets.

- The number of independent perturbation seeds is limited.

- The final OCaMo recovery result has an acknowledged numerical inference-path comparability limitation.

- The related shared-Roll selective calibration method has limited correction coverage.

- Conditional spatial detection recovery, including distance-specific OCaMo benefits, remains unverified.

- Real-world naturally occurring miscalibration and deployment-level gated fusion were not established.


The definitive implementation, source version, evidence path and environment for each component are listed in the [Experiment Index](experiment_index.md).
