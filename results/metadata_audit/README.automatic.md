# Metadata audit

## 1. Research question being audited
Can natural geometric/motion variation support separating sensor misalignment, geometric mismatch and later fusion degradation? This audit tests metadata support, not model performance.

## 2. Dataset and split
`v1.0-trainval`: **850 scenes, 64,386 instances, 1,166,187 annotations**. Final gating eligible: **True**.

## 3. Variable construction
See [methods](../../../docs/metadata_audit_methods.md) and `audit_config.json`. Position and velocities use the global/world horizontal frame, meters and seconds. Canonical Z is ego-origin horizontal distance at sample time. Object size order is W,L,H. World velocity uses central secants with actual elapsed time; endpoints are one-sided. Both adjacent gaps must be ≤ configured maximum. Ego positions/yaw are interpolated from LiDAR ego poses at annotation times without extrapolation; yaw is unwrapped before differences.
`v_rel = v_object − v_ego`; `r_hat = (p_object−p_ego)/Z`; radial = dot(v_rel,r_hat); |lateral| = absolute 2D cross product. This relative velocity excludes rotational apparent motion. Ego-frame bearing rate = signed_lateral/Z − ego_yaw_rate. Missing values remain NaN with validity fields.
Camera transform: R_cal^T [R_ego^T(p_world−t_ego)−t_cal]. Projection requires all box corners in front of the near plane and a nonzero convex-hull intersection with the image. Unclipped box-envelope width is not an occlusion-aware object width. Canonical projected columns use CAM_FRONT; all six cameras are in camera_projections.parquet.
Constructability: see [CSV](gates/constructability.csv); 0 variables with zero valid observations.

## 4. Motion diversity
- `object_speed`: P05=0, median=0.057573, P95=7.2537; missing=0.378%.
- `ego_speed`: P05=8.8142e-06, median=4.9628, P95=10.006; missing=0.000%.
- `relative_lateral_velocity`: P05=0.00038442, median=1.7483, P95=7.7498; missing=0.378%.
- `abs_ego_yaw_rate`: P05=2.803e-05, median=0.0071092, P95=0.20931; missing=0.000%.

Annotation-weighted summaries are not independent evidence. Sample-weighted ego distributions and trajectory/class summaries are separate in `motion/`.

## 5. Geometric diversity
Canonical distance median 29.328 m; P05–P95 7.467–73.874 m. Physical dimensions are conditioned on category; orientation is LOS-relative. See `geometry/`.

## 6. Visibility / sensing diversity
See `visibility/`: visibility by class/distance/scene, point support by class × distance, and per-camera FOV geometry. No raw sensor quality or per-camera occlusion is inferred from metadata.

## 7. Trajectory and scene independence
See `independence/`: annotation counts, trajectory length/duration, temporal neighbors, valid-velocity fraction, moving trajectories per scene and regime concentration. Unique counts across bins are not additive. C10_scene is uninformative when there are only 10 scenes.

## 8. Native sensor timing
Offsets pair six camera keyframes to the same sample’s LiDAR keyframe. Full sample_data chains supply sampling intervals (not just ~2 Hz keyframes). Signed/absolute seconds and raw microseconds are retained. Per-camera/scene distributions, Tukey diagnostics and lag-1..10 autocorrelation are saved; short windows do not prove periodicity.

## 9. Joint coverage
Each matrix has annotation, trajectory and scene versions, including zero cells. Bins are annotation-derived global quantiles with tied cutpoints collapsed. `bin_definitions.json` records exact edges. Concentration is in the corresponding coverage CSV.
- `distance_x_lateral_velocity`: 9/9 occupied, 9 SUPPORTED, 0 WEAK, 0 SPARSE.
- `size_x_lateral_velocity`: 9/9 occupied, 9 SUPPORTED, 0 WEAK, 0 SPARSE.
- `distance_x_visibility`: 12/12 occupied, 12 SUPPORTED, 0 WEAK, 0 SPARSE.
- `distance_x_lidar_support`: 9/9 occupied, 9 SUPPORTED, 0 WEAK, 0 SPARSE.
- `lateral_motion_x_ego_yaw_rate`: 9/9 occupied, 9 SUPPORTED, 0 WEAK, 0 SPARSE.
- `projected_width_x_lateral_velocity`: 9/9 occupied, 9 SUPPORTED, 0 WEAK, 0 SPARSE.

## 10. Confounding / identifiability
Preliminary diagnostic: **MANAGEABLE CONFOUNDING**. Qualifying same-scene, within-class coarse matched strata: `{'relative_lateral_velocity': 64, 'distance_xy': 70}`.
Spearman matrices use annotation, trajectory-median and scene-median levels; within-scene pair correlations are also saved. No iid p-values or causal claims. Coarse quantile overlap is not formal matching; absence only restricts this design.

## 11. Hypothetical mismatch coverage
`r_physical=|v_lateral|×dt/W`, dt ∈ {0.05,0.1,0.2,0.5} s. This excludes rotation and acceleration and is not actual degradation. Each dt/regime reports all three cluster counts in [coverage](mismatch_proxy/r_regime_coverage.csv).

## 12. Gate-by-gate assessment
Machine-readable decision and evidence: [gating_report.json](gating_report.json). Configurable project support thresholds are diagnostics, not universal laws. No suitability score is computed.
- constructability: **PASS**.
- motion_diversity: **VARIATION PRESENT**.
- geometry_diversity: **VARIATION PRESENT**.
- trajectory_independence: **SUPPORTED**.
- joint_coverage: **SUPPORTED**.
- identifiability: **MANAGEABLE CONFOUNDING**.
- mismatch_proxy_coverage: **SUPPORTED**.

## 13. Overall GO / CONDITIONAL GO / NO-GO
**CONDITIONAL GO**. Automatic hierarchical recommendation; requires research interpretation. Mini results never participate in the final decision.

## 14. Supported claims
- Metadata constructs the listed geometric and kinematic variables on validity-flagged rows.
- Observed descriptive variation and overlap can motivate restricted future tests, conditional on scene/trajectory support.

## 15. Unsupported / risky claims
- Actual fusion degradation/backfire or superiority of a normalization.
- Universal cross-scene/semantic-class generalization from observational coverage.
- Causal separation based solely on correlations or coarse matching.
- Observational metadata cannot establish fusion degradation, backfire, causal attribution or a normalization law.
- Clusters are nested and dependent; counts are not an effective independent sample size.
- Projection uses keyframe box position and camera-time ego pose; native object displacement is not compensated.
- Metadata visibility is not per-camera occlusion; LiDAR support is a provided count, not recomputed raw sensor quality.
- Velocity is a finite-difference estimate, not instantaneous ground truth; edges, gaps and annotation noise matter.
- Physical-width proxy ignores rotational image motion and orientation-dependent image scale.

## 16. Recommended next experiment
Inspect supported joint/matched regimes, prespecify scene-held-out comparisons and perturbation sweep; do not auto-deploy BEVFusion. Supplement/reformulate unsupported comparisons before acquisition.

No sensor-blob acquisition, GPU provisioning, BEVFusion, corruption injection or inference is triggered by this audit.
