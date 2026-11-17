# SOAC and OCaMo/LTO Baseline Audit

**Purpose:** establish SOAC as the scientific calibration reference and assess OCaMo/LTO as an implementation candidate. This audit does not claim to reproduce SOAC or to run OCaMo on nuScenes.

## 1. Scope

The project has two related but distinct questions:

- **Calibration:** can a method estimate or monitor camera–LiDAR extrinsic or timing error?
- **Perception robustness:** when a camera image is stale relative to a LiDAR keyframe, does fusion still help detection, and can temporal motion correction recover useful camera information?

SOAC and OCaMo/LTO address calibration. The BEVFusion offset experiment and temporal predictor address perception under stale visual content. Report calibration error separately from detector mAP/NDS and object-level backfire.

## 2. Scientific reference: SOAC

**Paper:** Quentin Herau et al., “SOAC: Spatio-Temporal Overlap-Aware Multi-Sensor Calibration using Neural Radiance Fields,” CVPR 2024, pp. 15131–15140.

- [CVPR page](https://openaccess.thecvf.com/content/CVPR2024/html/Herau_SOAC_Spatio-Temporal_Overlap-Aware_Multi-Sensor_Calibration_using_Neural_Radiance_Fields_CVPR_2024_paper.html)
- [Paper PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Herau_SOAC_Spatio-Temporal_Overlap-Aware_Multi-Sensor_Calibration_using_Neural_Radiance_Fields_CVPR_2024_paper.pdf)
- [arXiv full text](https://arxiv.org/abs/2311.15803)

### Method

SOAC alternates between training camera-specific implicit scene representations and optimizing sensor registration/calibration. It estimates extrinsic transformations and temporal parameters using camera observations and LiDAR scans. A visibility grid limits losses to regions observed across sensors; bounded translation and time corrections stabilize optimization.

This differs from the BEVFusion offset experiment: SOAC estimates calibration parameters from scene consistency, while the BEVFusion experiment selects camera frames at shifted timestamps, updates camera-to-LiDAR geometry, keeps LiDAR and labels fixed, and measures detector behavior.

### nuScenes protocol

The main spatiotemporal study uses four KITTI-360 sequences and three nuScenes sequences. The nuScenes sequences are **916, 410, and 417**, selected for nearby structures and speed variation. It uses the front camera, two front-diagonal cameras, and LiDAR; the front camera is the reference. It uses all LiDAR scans and every other camera image.

SOAC images are downscaled by 6 for nuScenes (MOISST by 2). Initial errors are 50 cm translation, 5° rotation on each axis, and 100 ms timing offset. Ten random seeds vary perturbation signs; results are summarized in box plots.

The authors state that nuScenes poses available to their pipeline were only SE(2), so they estimate LiDAR poses using KISS-ICP, then derive camera poses using the dataset extrinsics. This trajectory source is a material difference to record in any adaptation.

**Protocol caveat:** a separate dynamic-scene semantic-filtering experiment uses an analogous KITTI-360/nuScenes setup but removes temporal calibration and the initial time offset. That is a spatial-calibration experiment and must not be conflated with the main spatiotemporal experiment.

### Metrics and reported results

Calibration metrics are rotation error (degrees), translation error (centimeters), and time-offset error (milliseconds, where timing is optimized). These are not detection mAP/NDS.

For **KITTI-360**, SOAC reports aggregate errors (average over per-sensor medians) of 0.21° rotation, 5.24 cm translation, and 3.95 ms timing error. MOISST reports 2.24°, 56.34 cm, and 27.07 ms. These are **KITTI-360 results, not nuScenes results**.

Table 2 is spatial-only (no temporal error) and reports:

| Method | KITTI-360 rotation | KITTI-360 translation | Pandaset rotation | Pandaset translation |
|---|---:|---:|---:|---:|
| Pandey et al. | 11.8 ± 5.4° | 143 ± 109 cm | 15.4 ± 0.8° | 139 ± 17.5 cm |
| LCCNet | 1.9 ± 0.1° | 95.8 ± 7.7 cm | 14.3 ± 3.4° | 370 ± 11.6 cm |
| MOISST | 0.2 ± 0.1° | 10.0 ± 9.8 cm | 2.8 ± 2.3° | 56.4 ± 17.2 cm |
| SOAC | 0.3 ± 0.2° | 7.8 ± 3.5 cm | 1.3 ± 0.8° | 29.4 ± 13.6 cm |

NuScenes spatiotemporal results are plotted in Figure 5, not given as a matching numeric table in the main text. If exact nuScenes sequence values are needed, transcribe them from the figure/supplement. Do not substitute KITTI-360 or Pandaset values.

The separate dynamic-scene semantic-filtering experiment reports LiDAR calibration median error 0.41° / 7.79 cm with filtering versus 2.36° / 79.17 cm without; its all-sensor mean is 0.42° / 11.18 cm for SOAC versus 2.18° / 44.73 cm for MOISST. These are not nuScenes time-offset results.

### Limitations relevant to this project

- Constant-speed, straight-line motion can make temporal and extrinsic error indistinguishable. The paper shows apparently accurate poses but wrong calibration in such a case; speed variation is needed.
- Nearby structures and cross-sensor overlap improve constraints. Open scenes and long rays reduce translation precision.
- The NeRF does not model moving objects directly. Semantic filtering helps but can remove useful parked-vehicle structure.
- SOAC trains one NeRF per camera; the paper reports about 1 min 45 s per epoch for 3 nuScenes cameras and 8 min for 5 cameras on a GPU similar to RTX 3090.
- SOAC relies on a reference sensor trajectory; the KISS-ICP choice on nuScenes must be documented.

## 3. Implementation candidate: OCaMo/LTO

**Paper:** J. Moravec and R. Šára, “Online Camera–LiDAR Calibration Monitoring and Rotational Drift Tracking,” IEEE T-RO, vol. 40, pp. 1527–1545, 2024, DOI 10.1109/TRO.2023.3347130.

- [Official code](https://github.com/moravecj/OCaMo)
- [Project page and supplement](https://cmp.felk.cvut.cz/~moravj34/ocamo/)

OCaMo uses adaptive online stochastic optimization with memory; LTO uses fixed-grid search per frame without memory. Both target online extrinsic-calibration monitoring/tracking.

The released Python examples run on a supplied CARLA snippet and demonstrate validity monitoring and synthetic rotational drift tracking. Their successful run verifies the examples and dependencies in this container, but does **not** establish nuScenes compatibility or detector improvement. The Python inference folder has no ready-made nuScenes adapter. The repository is licensed CC BY-NC 4.0; preserve attribution and check the terms before redistributing modified code.

## 4. nuScenes data suitability

The existing nuScenes data/metadata provide images, LiDAR scans, calibrated sensor transforms, poses, timestamps, and scene/sample records. These support injecting known synthetic temporal or extrinsic perturbations and measuring recovery against the known injected error.

They do not provide ground truth for naturally occurring calibration drift. Describe results as synthetic-perturbation recovery, not measurement of real drift.

A SOAC comparison must document the three-camera subset, selected sequences, LiDAR pose source, subsampling, downscaling, and whether temporal calibration is enabled. The current six-camera BEVFusion setup and its pose handling differ unless deliberately matched.

## 5. Baseline acceptance protocol

### Gate A — scientific audit
- Record the above settings/results with dataset labels.
- Transcribe nuScenes Figure 5/supplement values if exact values are needed.
- Keep the main spatiotemporal and dynamic spatial-only experiments separate.

### Gate B — OCaMo/LTO example check
- Record commands, package versions, exit status, and plots for the CARLA examples.
- Treat this only as a code smoke check.

### Gate C — nuScenes adapter pilot
- Map nuScenes images, LiDAR, intrinsics, extrinsics, timestamps, poses, coordinate frames, and units to the exact OCaMo input format.
- Validate a few frames and a zero-perturbation sequence.
- Inject known rotational perturbations and compare estimates with the known errors; report per-sequence rotation error, validity detection, runtime, and failures.
- Claim timing recovery only if the selected method and adapted objective actually estimate timing. The released rotational-tracking demo does not establish that.

### Gate D — downstream BEVFusion
If Gate C succeeds, apply estimated corrections through the existing camera-to-LiDAR transform path. Compare clean calibration, injected error, corrected calibration, oracle correction, and LiDAR-only. Report calibration accuracy separately from detector and object-level metrics.

## 6. Recommended interpretation

Use SOAC as the scientific calibration reference and OCaMo/LTO as a candidate monitoring implementation, subject to a successful nuScenes adapter. Keep temporal object-motion prediction/correction as a separate robustness method. The approaches can complement each other: first control or estimate calibration, then test whether motion correction recovers camera evidence from stale frames.

## References

1. Herau et al., SOAC, CVPR 2024: https://arxiv.org/abs/2311.15803
2. SOAC CVF page: https://openaccess.thecvf.com/content/CVPR2024/html/Herau_SOAC_Spatio-Temporal_Overlap-Aware_Multi-Sensor_Calibration_using_Neural_Radiance_Fields_CVPR_2024_paper.html
3. Moravec and Šára, OCaMo/LTO: https://cmp.felk.cvut.cz/~moravj34/ocamo/
4. Official OCaMo repository and license: https://github.com/moravecj/OCaMo

