# Metadata audit

## 1. Research question being audited
Can natural geometric/motion variation support separating sensor misalignment, geometric mismatch and later fusion degradation? This audit tests metadata support, not model performance.

## 2. Dataset and split
`v1.0-trainval`: **850 scenes, 64,386 instances, 1,166,187 annotations**. Final gating eligible: **True**.


Official split membership independently checked: **700 train + 150 val**, 34,149 samples. Source JSON row counts exactly match 64,386 instances and 1,166,187 annotations. See `source_and_scope_validation.json`. Mini numerical checks and the full run are separate; mini contributes no suitability evidence.

## 3. Variable construction
See [methods](../../../docs/metadata_audit_methods.md) and `audit_config.json`. Position and velocities use the global/world horizontal frame, meters and seconds. Canonical Z is ego-origin horizontal distance at sample time. Object size order is W,L,H. World velocity uses central secants with actual elapsed time; endpoints are one-sided. Both adjacent gaps must be ≤ configured maximum. Ego positions/yaw are interpolated from LiDAR ego poses at annotation times without extrapolation; yaw is unwrapped before differences.
`v_rel = v_object − v_ego`; `r_hat = (p_object−p_ego)/Z`; radial = dot(v_rel,r_hat); |lateral| = absolute 2D cross product. This relative velocity excludes rotational apparent motion. Ego-frame bearing rate = signed_lateral/Z − ego_yaw_rate. Missing values remain NaN with validity fields.
Camera transform: R_cal^T [R_ego^T(p_world−t_ego)−t_cal]. Projection requires all box corners in front of the near plane and a nonzero convex-hull intersection with the image. Unclipped box-envelope width is not an occlusion-aware object width. Canonical projected columns use CAM_FRONT; all six cameras are in camera_projections.parquet.
Constructability: see [CSV](gates/constructability.csv); 0 variables with zero valid observations.


Object and relative velocities are valid for **1,161,782 / 1,166,187 annotations (99.6223%)**, spanning 62,083 instances and all 850 scenes. The 4,405 missing values remain NaN. Other critical metadata variables have full constructability on their respective populations. CAM_FRONT projected width is available for **331,904 annotations (28.4606%)**, 32,252 trajectories and 850 scenes; this is FOV-conditioned availability, not a 71.5% metadata failure rate.

## 4. Motion diversity
- `object_speed`: P05=0, median=0.057573, P95=7.2537; missing=0.378%.
- `ego_speed`: P05=8.8142e-06, median=4.9628, P95=10.006; missing=0.000%.
- `relative_lateral_velocity`: P05=0.00038442, median=1.7483, P95=7.7498; missing=0.378%.
- `abs_ego_yaw_rate`: P05=2.803e-05, median=0.0071092, P95=0.20931; missing=0.000%.

Annotation-weighted summaries are not independent evidence. Sample-weighted ego distributions and trajectory/class summaries are separate in `motion/`.


Full-data sanity diagnostics retain rather than delete extremes: maximum object speed 48.825 m/s, ego speed 18.392 m/s and |ego yaw rate| 0.6663 rad/s. Nominally stationary barrier/cone speed P95 is 0.2152 m/s, indicating annotation/estimation noise matters near zero. `motion/high_speed_diagnostics.csv` and examples identify records to inspect before interpreting fast-motion tails. Object-speed and lateral-relative-speed medians differ markedly (0.0576 versus 1.7483 m/s); ego motion is a major contributor.

## 5. Geometric diversity
Canonical distance median 29.328 m; P05–P95 7.467–73.874 m. Physical dimensions are conditioned on category; orientation is LOS-relative. See `geometry/`.


Among valid CAM_FRONT projections, **50,942** are partially outside the image and **3,295** have unclipped width >1,600 px. Do not use such widths as measured visible box sizes. Keep boundary-conditioned sensitivity analyses for a later image-scale normalization.

## 6. Visibility / sensing diversity
See `visibility/`: visibility by class/distance/scene, point support by class × distance, and per-camera FOV geometry. No raw sensor quality or per-camera occlusion is inferred from metadata.

## 7. Trajectory and scene independence
See `independence/`: annotation counts, trajectory length/duration, temporal neighbors, valid-velocity fraction, moving trajectories per scene and regime concentration. Unique counts across bins are not additive. C10_scene is uninformative when there are only 10 scenes.


Using the separately declared |yaw rate| thresholds, high-yaw (>0.1 rad/s) coverage is **116,327 annotations / 16,462 trajectories / 393 scenes**. It is not confined to one or two scenes. `gates/support_threshold_sensitivity.csv` varies the scene/trajectory requirements, without presenting any cutoff as a statistical law.

## 8. Native sensor timing
Offsets pair six camera keyframes to the same sample’s LiDAR keyframe. Full sample_data chains supply sampling intervals (not just ~2 Hz keyframes). Signed/absolute seconds and raw microseconds are retained. Per-camera/scene distributions, Tukey diagnostics and lag-1..10 autocorrelation are saved; short windows do not prove periodicity.


Camera median offsets (camera minus LiDAR, ms): BACK −10.749; BACK_LEFT −0.876; BACK_RIGHT −20.302; FRONT −35.862; FRONT_LEFT −43.472; FRONT_RIGHT −27.895. After subtracting each camera median, P05–P95 residuals are −2.079 to +1.582 ms; the pooled offset range is not mainly random broad jitter. Rare extremes remain (−117.877 to +49.992 ms). This supports camera-dependent nominal offsets plus residual variation, not perfect synchronization or a demonstrated periodic pattern.

## 9. Joint coverage
Each matrix has annotation, trajectory and scene versions, including zero cells. Bins are annotation-derived global quantiles with tied cutpoints collapsed. `bin_definitions.json` records exact edges. Concentration is in the corresponding coverage CSV.
- `distance_x_lateral_velocity`: 9/9 occupied, 9 SUPPORTED, 0 WEAK, 0 SPARSE.
- `size_x_lateral_velocity`: 9/9 occupied, 9 SUPPORTED, 0 WEAK, 0 SPARSE.
- `distance_x_visibility`: 12/12 occupied, 12 SUPPORTED, 0 WEAK, 0 SPARSE.
- `distance_x_lidar_support`: 9/9 occupied, 9 SUPPORTED, 0 WEAK, 0 SPARSE.
- `lateral_motion_x_ego_yaw_rate`: 9/9 occupied, 9 SUPPORTED, 0 WEAK, 0 SPARSE.
- `projected_width_x_lateral_velocity`: 9/9 occupied, 9 SUPPORTED, 0 WEAK, 0 SPARSE.


Distance × lateral-motion cells contain **9,144–28,787 trajectories** and **769–850 scenes** each. The largest single-scene contribution among these cells is **1.70%**. Pooled coverage is broad, but the class × motion × distance table identifies rare-class exceptions.

## 10. Confounding / identifiability
Preliminary diagnostic: **MANAGEABLE CONFOUNDING**. Qualifying same-scene, within-class coarse matched strata: `{'relative_lateral_velocity': 64, 'distance_xy': 70}`.
Spearman matrices use annotation, trajectory-median and scene-median levels; within-scene pair correlations are also saved. No iid p-values or causal claims. Coarse quantile overlap is not formal matching; absence only restricts this design.


Important descriptive Spearman pairs: distance–lateral motion **−0.138**, distance–LiDAR count **−0.652**, width–distance **+0.209**, |yaw rate|–lateral motion **+0.356**, distance–projected width **−0.477**. None exceeds the configured 0.7 flag, but this is not evidence of no confounding. In particular, **87.16%** of the middle-width bin consists of cars. The 64/70 qualifying matched strata include nine car strata in each direction. Ambulance cells are all SPARSE; several mobility and other rare categories have limited support. Restrict claims to supported classes and matched regimes.

## 11. Hypothetical mismatch coverage
`r_physical=|v_lateral|×dt/W`, dt ∈ {0.05,0.1,0.2,0.5} s. This excludes rotation and acceleration and is not actual degradation. Each dt/regime reports all three cluster counts in [coverage](mismatch_proxy/r_regime_coverage.csv).


For **vehicle.car**, large normalized mismatch (`r>=1`) has the following class-specific support:

| dt | N_ann | N_traj | N_scene |
|---|---:|---:|---:|
| 50 ms | 0 | 0 | 0 |
| 100 ms | 567 | 378 | 117 |
| 200 ms | 9,750 | 3,366 | 559 |
| 500 ms | 137,782 | 18,110 | 828 |

At 50 ms the pooled r>=1 band exists (6,865 / 2,382 / 359), but **none of it is vehicle.car**. This is precisely why pooled mismatch coverage cannot justify every class-specific experiment. See `mismatch_proxy/r_regime_by_class.csv`.

## 12. Gate-by-gate assessment
Machine-readable decision and evidence: [gating_report.json](gating_report.json). Configurable project support thresholds are diagnostics, not universal laws. No suitability score is computed.
- constructability: **PASS**.
- motion_diversity: **VARIATION PRESENT**.
- geometry_diversity: **VARIATION PRESENT**.
- trajectory_independence: **SUPPORTED**.
- joint_coverage: **SUPPORTED**.
- identifiability: **MANAGEABLE CONFOUNDING**.
- mismatch_proxy_coverage: **SUPPORTED**.


Research review keeps the automatic recommendation but adds a stricter class/dt interpretation. Gates 1–4 and the pooled mismatch gate have broad support. Gate 5 requires within-class/quality-controlled comparisons; Gate 6 must be read conditionally on semantic class and dt. `research_review.json` preserves explicit evidence and restrictions. The automatic report is retained separately for transparency.

## 13. Overall GO / CONDITIONAL GO / NO-GO

**CONDITIONAL GO — proceed with a restricted, prespecified temporal-misalignment experiment.** The data do not falsify the feasibility of the central geometric question, but do not justify unrestricted cross-class or cross-scene normalization claims. This is research judgment grounded in the diagnostics, not a suitability score.

## 14. Supported claims

- All structurally essential requested variables are constructible on the full official 850-scene trainval split; object/relative velocity remains missing on 4,405 annotations rather than imputed.
- Distance/lateral-motion comparisons have broad scene and trajectory support: all nine global tertile cells meet the declared project support rules.
- Within-class coarse comparisons can hold distance/width approximately fixed while lateral motion varies, and conversely; overlap is evidence for designing comparisons, not a causal estimate.
- A class-conditioned temporal perturbation study is feasible. For vehicle.car, 100–200 ms supports all four specified physical-proxy bands; 50 ms is a lower-mismatch control axis.
- Native sensor offsets are predominantly camera-dependent with additional scene variation and outliers; actual timestamps should define future perturbations.

## 15. Unsupported / risky claims

- An existing fusion failure/backfire boundary, accuracy degradation curve, or validated cross-scene normalization law.
- Uniform large-mismatch coverage at 50 ms for vehicle.car, or pooled coverage generalized to all semantic classes.
- Causal separation of size, motion, distance and sensing quality from a correlation matrix or coarse quantile matching alone.
- Treating 1,166,187 annotations as independent trials, or treating counts of trajectories/scenes as a formally estimated effective sample size.
- Interpreting camera FOV projection as unoccluded camera visibility, or physical width W as an orientation-independent projected width.
- Treating the 0.5 s first-order proxy or finite-difference velocity as exact instantaneous geometry.
- Pooled mismatch coverage does not transfer to every class/dt. In vehicle.car, 50 ms has no r>=1 observations; see class-conditioned coverage.
- Physical size is semantically confounded: the middle-width bin is dominated by vehicle.car. Use within-class comparisons and observed overlapping support, not pooled size effects.
- CAM_FRONT projected scale is valid only on its geometric-FOV subset; near-plane rejection and partial image boundaries create selection. Large unclipped box envelopes are not visible object widths.
- Distance and LiDAR point support are dependent (descriptive pooled Spearman about -0.652); control observation quality when comparing geometric regimes.
- Relative lateral motion is strongly influenced by ego translation and is not equivalent to moving-object speed. Ego rotation needs a separate axis or a full projection-based model.
- Rare semantic classes do not share the broad pooled support. Emergency ambulance and some mobility classes cannot support the same broad claims under the configured design thresholds.

## 16. Recommended next experiment

Proceed with a restricted experimental design, not automatic full sensor acquisition: start with well-supported classes such as vehicle.car and adult pedestrians; prespecify distance × lateral-motion strata within class and width strata, balance visibility/LiDAR support, and separate high-yaw scenes. Use 50/100/200 ms initially, with 500 ms as a clearly approximate stress test. Reserve held-out scenes (prefer also log-level separation), cluster summaries by trajectory/scene, and check clipped versus unclipped camera scale before testing normalization. Download only the sensor subsets needed for that prespecified design, then obtain explicit authorization for GPU/BEVFusion deployment. Current evidence does not require switching to Waymo/AV2; supplement only if a targeted unsupported class/regime or external generalization claim is required.

No BEVFusion, GPU provisioning, corruption or inference has been executed. No trainval Camera/LiDAR blobs were required for this audit. All row-level Parquet tables and large trajectory tables remain at `/workspace/results/metadata_audit/`; the repository contains code, configuration and small derived summaries. See `artifact_inventory.json` for persistent artifacts.
