# Curated Project Results

This directory contains selected presentation figures from the completed Camera–LiDAR Backfire research.

The figures are copied from existing experimental outputs. They are not newly generated measurements.

## Figures

|File|Research component|Source|
|---|---|---|
|[spatial_recovery.png](spatial_recovery.png)|Spatial rotation robustness and integrated OCaMo–BEVFusion recovery|Locally recovered final presentation figure, November 18, 2026|
|[temporal_overview.png](temporal_overview.png)|Temporal misalignment and ego-consistency experiments|`jhe-sudo/temporal-misalignment`, commit `2e8dd75`|
|[conditional_displacement.png](conditional_displacement.png)|Object-level temporal backfire versus normalized displacement|`nuo-ft-conditional-threshold`, commit `58b49c5`|

All copied images must retain their original source bytes and have verified SHA-256 checksums.

## Selected Quantitative Results

### Spatial rotation robustness

Five Gaussian perturbation levels (σ = 1°–5°), five seeds each, 6,019 validation frames per condition.

The published original Spatial experiment found mean fusion mAP below the LiDAR-only reference at σ = 4° and 5°, while acknowledging seed-level uncertainty.

[Original full results](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/5e8bf5701ad4db1da13d73eabd361eb10668a10e/rotation_robustness/README.md)

### Online calibration recovery

|Quantity|Reported result|
|---|---|
|Corrected conditions|25/25|
|Corrected prediction frames|150,475|
|Camera trajectories with reduced rotational error|76.6%|
|Average Corrected − Perturbed mAP|+0.1845 percentage points|
|Average Corrected − Perturbed NDS|+0.1112 percentage points|
|Corrected fusion at σ = 4° and 5°|Mean remains below LiDAR-only|

The relatively small detection differences are subject to the documented historical inference-path numerical-comparability limitation.

### Temporal misalignment

The completed temporal synthesis found that the effects of sensor staleness depend strongly on modality and consistency of geometric metadata. Naive LiDAR staleness is especially damaging, while ego-consistent camera delivery retains useful fusion performance over much larger tested delays.

[Temporal synthesis](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/2e8dd75aecf6be6f2de8249f5cfce02c35fb8167/results/temporal_misalignment/temporal_misalignment_synthesis_en.md)

### Conditional backfire

Object-level backfire onset depends strongly on normalized lateral displacement, with earlier miss-rate crossover in higher-displacement groups.

[Conditional analysis](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/58b49c53e3bf452ad6b87438c5a94b72ad852cac/scripts/temporal/conditional_threshold/README.md)

## Scope and Provenance

This directory is a **presentation-level summary**, not a complete prediction archive.

The complete implementation, original metrics and protocol-specific limitations are preserved in their respective experiment branches and reports.

Do not interpret the curated figures as a common evaluation population across the Spatial and Temporal research tracks.

Return to the [Project Homepage](../../README.md) or the [Experiment Index](../../docs/experiment_index.md).
