# Experiment and Branch Index

This document is the navigation map for the Camera–LiDAR Backfire repository.

The project contains two principal research tracks: **Spatial Misalignment** and **Temporal Misalignment**.

Experimental branches remain separate to preserve their implementation history, environment requirements and ownership boundaries.

**Important:** The root `main` branch is the project homepage and historical infrastructure baseline. It is not a unified executable implementation of all research tracks.

All references below use fixed commit hashes reviewed on November 18, 2026.

## 1. Principal Research Branches

|Branch|Purpose|Primary entry point|
|---|---|---|
|`noah/robustness-experiments`|Completed Spatial rotation robustness and related conditional analysis|[Rotation robustness README](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/5e8bf5701ad4db1da13d73eabd361eb10668a10e/rotation_robustness/README.md)|
|`jhe-sudo/temporal-misalignment`|Temporal misalignment, ego-consistent variants, joint staleness and mechanism analysis|[Complete temporal synthesis](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/2e8dd75aecf6be6f2de8249f5cfce02c35fb8167/results/temporal_misalignment/temporal_misalignment_synthesis_en.md)|
|`nuo-ft-conditional-threshold`|Object-level conditional temporal backfire thresholds|[Conditional threshold README](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/58b49c53e3bf452ad6b87438c5a94b72ad852cac/scripts/temporal/conditional_threshold/README.md)|

The conditional-threshold branch extends the temporal study; it is not a third independent principal track.

## 2. Calibration and Supporting Branches

|Branch|Purpose|Primary entry point|
|---|---|---|
|`noah/ocamo-baseline`|OCaMo method reconstruction, baseline experiments and scientific limitations|[Full-method results](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/85f489a840a65bacbbe139b9781fc062faac58e0/docs/ocamo/FULL_METHOD_RESULTS.md)|
|`noah/ocamo-experiments`|Earlier OCaMo working checkpoint and implementation handoff|[Historical handoff](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/95c3fa452a749a4d88a788413ea8af1ac137a517/docs/ocamo/HANDOFF.md)|
|`noah/environment-baseline`|Environment, BEVFusion deployment and reproducibility groundwork|[BEVFusion deployment report](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/f9acebc5b39076138b656315476a667f339d6109/docs/bevfusion_deployment_report.md)|
|`refactor/windows-local-setup`|Windows environment adaptation and complementary six-camera shared-Roll selective calibration experiments|[Calibration study README](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/6253452f0131e19791b1a3deee566ccede2ca1b8/README.md)|
|`temporal_misalignment`|Earlier temporal working branch, retained for historical reference|[Historical branch snapshot](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/f2019f118ef94ad492a8a699ab1ce70b84392e69)|

The `main` branch retains the initial metadata audit and shared repository infrastructure. Its reviewed pre-publication commit was `bd6625e292995418e96490dd001dd7f3e01af4a9`.

**Total: nine existing branches including `main`.**

Historical branches should not be mistaken for the latest scientific status merely because they have similar names.

## 3. Final Integrated OCaMo–BEVFusion Results

The completed November 18, 2026 integrated experiment contains:

- 25 conditions (five noise levels × five seeds).

- 150,475 corrected BEVFusion prediction frames.

- Full detection and early/late evaluations.

- Final scientific report, plots and numerical tables.

- An acknowledged historical-versus-current inference numerical-comparability limitation.


The selected final presentation figure is preserved on `main` at:

[Spatial Recovery Figure](../results/project_summary/spatial_recovery.png)

The complete integrated experiment report and large prediction artifacts were originally recovered to local and persistent project storage, not committed as part of the earlier OCaMo baseline branch.

This index must not claim that the earlier `FULL_METHOD_RESULTS.md` contains the final integrated BEVFusion evaluation.

## 4. Reproducibility by Component

### Spatial rotation experiment

[Full results and instructions](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/5e8bf5701ad4db1da13d73eabd361eb10668a10e/rotation_robustness/README.md)

[Per-condition metrics](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/5e8bf5701ad4db1da13d73eabd361eb10668a10e/rotation_robustness/results/metrics_all_conditions.csv)

[Five-seed statistics](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/5e8bf5701ad4db1da13d73eabd361eb10668a10e/rotation_robustness/results/metrics_by_sigma.csv)

### Temporal experiments

[Full synthesis and report inventory](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/2e8dd75aecf6be6f2de8249f5cfce02c35fb8167/results/temporal_misalignment/temporal_misalignment_synthesis_en.md)

[Temporal experimental scripts](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/2e8dd75aecf6be6f2de8249f5cfce02c35fb8167/scripts/temporal)

### Conditional temporal thresholds

[Methods and results](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/58b49c53e3bf452ad6b87438c5a94b72ad852cac/scripts/temporal/conditional_threshold/README.md)

### OCaMo calibration reconstruction

[Full-method results and limitations](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/85f489a840a65bacbbe139b9781fc062faac58e0/docs/ocamo/FULL_METHOD_RESULTS.md)

### Complementary selective calibration

[Experiment methods and frozen-gate evaluation](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/6253452f0131e19791b1a3deee566ccede2ca1b8/README.md)

### Initial metadata audit

[Main-branch audit](../results/metadata_audit/README.md)

[Audit methodology](metadata_audit_methods.md)

## 5. Existing Pull Requests

The following pull requests were open at the reviewed snapshot:

- [PR #3 — Temporal misalignment](https://github.com/Strange-Noah/Camera-LiDAR-backfire/pull/3)

- [PR #4 — Conditional threshold](https://github.com/Strange-Noah/Camera-LiDAR-backfire/pull/4), based on the temporal branch

- [PR #5 — Spatial calibration and Windows setup](https://github.com/Strange-Noah/Camera-LiDAR-backfire/pull/5)


These pull requests are provided for navigation only. Their inclusion here is **not** an approval to merge them.

In particular, PR #5 includes a large embedded model/framework checkout and must not be treated as a documentation-only merge.

## 6. Repository and Artifact Policy

- Keep experimental branches independent unless integration is separately reviewed.

- Use fixed commits when citing numerical results or reproduction procedures.

- Do not describe historical calibration diagnostics as final detection-recovery results.

- Do not compare metrics across different nuScenes subsets without qualification.

- Do not commit raw nuScenes sensor data, pretrained weights, credentials or large prediction files.

- Preserve attribution and original third-party licenses.

- Use `main` as the unified research presentation and navigation entry point, not as an implied unified runtime environment.
