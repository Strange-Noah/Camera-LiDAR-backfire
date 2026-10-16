# Metadata audit methods and interpretation

## Scope and execution

This pipeline uses official nuScenes metadata only. Mini is development/verification data; **only an exactly 850-scene v1.0-trainval run can issue the final recommendation**. It never downloads data, starts a GPU, installs BEVFusion or runs corruption/inference. Acquisition is a separate explicit script. Raw JSON, Parquet tables, SQLite indexes and full logs stay on Runpod; code/configuration, small summary tables and figures are versioned in GitHub.

```bash
./scripts/setup_metadata_env.sh
PYTHONPATH=src /opt/venvs/nuscenes-metadata/bin/python -m unittest discover -s tests -v
./scripts/run_metadata_audit.sh --version v1.0-mini --output /workspace/results/metadata_audit_mini
/opt/venvs/nuscenes-metadata/bin/python scripts/validate_audit_mini.py --tables /workspace/results/metadata_audit_mini/tables
# Only after mini numerical validation succeeds:
python3.12 scripts/acquire_nuscenes_trainval_metadata.py
./scripts/run_metadata_audit.sh --version v1.0-trainval --output /workspace/results/metadata_audit
```

SQLite is a disposable streaming index under `/tmp/nuscenes-audit-index`; original JSON remains on the Network Volume. Individual scenes are constructed and appended as Parquet row groups. Completed outputs cannot be overwritten by the entry point. Use a fresh output directory for a new version/configuration. If analysis alone failed after successful table construction, `--analyze-only` can resume that incomplete run. `RUN_COMPLETE.json` is written only after all analyses/report generation finish. Mini's separate `numerical_validation.json` confirms reference tests.

## Canonical populations, frames and units

- `objects.parquet`: one row per annotation token at a sample/keyframe, nested in instance and scene. World positions in meters; timestamps are int64 microseconds. W,L,H are nuScenes annotation `size=[width,length,height]`, with box local x along length and local y along width. Quaternion order is w,x,y,z. Yaw is atan2(R[1,0],R[0,0]).
- `trajectories.parquet`: one row per instance. Instance scene membership is checked, not assumed. End tokens and declared annotation counts are checked. `track_position` is zero-based; `track_length` is annotation count, not elapsed time. `duration` is seconds. Visibility range is ordinal max token minus min token, not a visibility-percentage interval.
- `scenes.parquet`: scene duration, sample/annotation/instance counts, category counts, ego motion, moving-trajectory counts and geometric coverage. `ego_yaw_change` is signed unwrapped net change (rad); `ego_yaw_total_variation` is total absolute incremental change. Mean ego speed is sample-weighted, distance travelled follows the sampled ego path. Moving means speed > configured 0.5 m/s, a descriptive design threshold, not ground truth motion labeling.
- `sensor_timing.parquet`: one row per keyframe and camera, paired with that sample's LIDAR_TOP keyframe. `sample_timestamp`, `camera_timestamp`, `lidar_timestamp` are raw microseconds; all differences have explicit `_us` and `_s` columns. Object trajectory counts do not apply to this population and are reported as null, not fabricated by duplicating pairs across annotations.
- Extra canonical tables: `ego_samples.parquet` (one ego estimate per sample), `sensor_intervals.parquet` (one row per camera/LiDAR sample_data observation, including non-keyframes), `camera_projections.parquet` (one annotation-camera pair for all six cameras). Per-camera projection tuples are not six independent objects.

All translational velocities and relative-vector calculations use the **same global/world XY frame**. Canonical Z is `distance_xy` from ego origin to object center at annotation time; `distance_3d` is also retained. This Z is not optical depth. A future camera-specific normalization must distinguish ego range from camera optical Z.

## Velocity and time alignment

For object trajectories, the central estimate is the secant `(p_next-p_prev)/(t_next-t_prev)`, using actual intervals, including unequal spacing. Endpoints use `(p_next-p_current)/dt` or `(p_current-p_prev)/dt`. A singleton yields NaN. Every adjacent gap in the stencil must be positive and ≤1.5 s by default. This is stricter than the devkit total-central-span limit: two uneven gaps whose total is acceptable can still be rejected. No nearest-neighbor substitution or zero-fill is used.

Ego world positions and unwrapped yaw are interpolated from all available LiDAR ego poses to exact sample timestamps. No extrapolation beyond the observed pose interval is allowed. The same central/one-sided rule then differentiates these sample-time poses. This yields a keyframe-scale smoothed velocity estimate, not instantaneous CAN-bus ground truth. Both object and ego velocity estimates have validity, temporal-neighbor, method and span fields. Annotation noise, angular acceleration and finite stencil width remain limitations. Unwrapped ego yaw is differentiated before any wrap used for stored orientation.

`v_rel=v_object-v_ego`; `r_hat=(p_object-p_ego)/||p_object-p_ego||_XY`; `v_radial=v_rel·r_hat`. Positive radial means receding. Signed lateral is the 2D cross product `r_hat_x*v_rel_y-r_hat_y*v_rel_x`; its absolute value is the requested `relative_lateral_velocity`. For nonzero range, radial²+lateral²=relative_speed². Zero range yields invalid radial/lateral values. `velocity_valid` refers to object velocity; `relative_velocity_valid` additionally needs ego velocity and nonzero range.

This world-vector subtraction does **not** include ego rotation. The additional ego-frame horizontal bearing-rate diagnostic is `signed_lateral/Z - ego_yaw_rate` (rad/s); it is still not a calibrated pixel displacement. Ego yaw rate is independently included in joint coverage. No silent claim that `v_perp` captures all image motion is made.

## Projection and observability

For a world corner p, camera coordinates are:

`p_cam = R_cal^T (R_ego^T (p_world-t_ego) - t_cal)`.

Calibration is sensor-to-ego, pose is ego-to-world. Full 3D quaternion rotations are used, including roll/pitch; yaw-only approximations are not used for projection. Corners are projected by K and divided by camera z. All corners must have z >0.1 m; crossing the near plane is explicitly recorded and rejected, not clamped into a huge artificial width. The convex hull of projected corners must have positive-area overlap with `[0,width]×[0,height]`.

`camera_visible` means **geometric FOV intersection only**, not unoccluded visibility. It is conservatively false when near-plane geometry is rejected; `projection_depth_valid` and `near_plane_crossing` distinguish that reason. `projection_valid` requires the above conditions. Width/height are the full, **unclipped** axis-aligned projected 3D-box envelope; `partially_outside_image` records clipping at the image boundary. Invalid/out-of-FOV projections have NaN width/height. Canonical object columns refer to CAM_FRONT; all cameras remain in the separate pair table.

Boxes remain at annotation/sample time, while the camera's own ego pose and calibration are used. This follows the official devkit keyframe-box transform convention. `camera_minus_sample_dt_s` records the residual temporal mismatch; dynamic object position is **not** silently propagated to camera time. Thus projected width is an approximate geometric scale, not a measured image box. A later image-level study must handle object-time interpolation, occlusion and actual image content explicitly.

## Hierarchical diagnostics

Each important bin/joint cell retains `(N_ann,N_traj,N_scene)` plus top-1/3/10 scene and top-1/10 trajectory fractions. Unique counts in different cells overlap and must not be summed as independent observations. Scene/trajectory count is not a computed statistical effective sample size. No iid confidence intervals or p-values are emitted.

Initial bins are global **annotation-weighted** tertiles. Duplicate cutpoints collapse; identical values are never arbitrarily split to force three equally populated bins. Edges and labels are stored in `bin_definitions.json`. Missing observations are excluded from applicable bins and separately reported in constructability. Visibility uses four ordinal categories (v0–40, v40–60, v60–80, v80–100); it is not a continuous percentage. Physical yaw thresholds (0.02,0.1 rad/s) and moving speed are separately labeled project conventions.

Spearman is provided at annotation, trajectory-median and scene-median levels, with within-scene pair diagnostics. Different aggregation levels answer different questions and are not interchangeable. Pearson is omitted because pooled heavy tails/mixtures and repeated observations give no reason to privilege a linear dependence summary. Class dominance uses raw category labels without silently merging rare classes.

Matched support conditions on category and two coarse quantile bins, asking whether the target's global low/high bins both occur in common scenes. Both extremes must have the configured trajectory support **within those common scenes**. This checks overlap, not a causal effect. Coarse bins may leave residual confounding. Failure means this particular comparison design lacks support; it does not mathematically prove no alternative match can exist.

## Timing diagnostics

Native offset is camera timestamp minus paired LiDAR timestamp, in raw microseconds and seconds. Inter-observation intervals follow actual sample_data prev/next links, including sweeps. Linked channel, reciprocal token, positive interval and scene membership are checked. The missing endpoint interval is NaN. Each next interval is counted once; a LiDAR interval is not counted six times in the sampling-interval table.

Per-camera/scene offset distributions, signed/absolute quantiles, camera and scene medians, within-camera residuals and keyframe lag autocorrelations are retained. Sampling outliers use descriptive Tukey fences, not a sensor failure definition. Scene windows are short, so autocorrelation is evidence to inspect, not an automatic periodicity claim.

## Proxy and hierarchical gates

`r_physical=abs(v_perp)*delta_t/W`, for delta_t = 0.05,0.1,0.2,0.5 s. This is a first-order translation/physical-width proxy. It ignores acceleration, ego/object rotation, view-dependent width and occlusion. In particular 0.5 s is not guaranteed to be locally linear. No image displacement proxy or fusion degradation is claimed. The four r bands are explicit user-requested hypothetical regimes, not performance thresholds.

Default project support diagnostics: SUPPORTED requires ≥30 trajectories, ≥10 scenes and top-scene annotation share ≤0.30. WEAK requires ≥5 trajectories and ≥3 scenes; otherwise SPARSE. Matching requires ≥3 common scenes and ≥5 trajectories per extreme there. Strong Spearman flag is |rho|≥0.7; class-dominance flag is ≥0.8. **These are editable project-design choices, not universal statistical significance cutoffs.** Conservative joint coverage considers zero cells and every requested primary bin; a few sparse cells can restrict a claim without invalidating the entire dataset.

The hierarchy is constructability → variation → independent coverage → joint support → confounding/overlap → proxy support. Structural absence can block the proposed formulation; missingness or a high correlation alone does not. Mini's overall field is always NOT ASSESSED. An eligible full run emits an automatic recommendation and explicit supported/unsupported claims, never a suitability score or an instruction that automatically deploys models. The final written interpretation must consider diagnostic evidence, not blindly accept a threshold label.

## What metadata can and cannot settle

All requested canonical variables, provided LiDAR/radar counts, categorical visibility, native timing and geometric box projections can be constructed from metadata alone when their local validity conditions hold. No additional trainval sensor blobs are needed for these audit statistics. Actual image visibility/occlusion, blur, exposure, LiDAR returns/noise, accurate camera-time dynamic object geometry and any model performance require additional evidence. Only request targeted sensor blobs if a later validation specifically needs them.

## Primary references

- [Official nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md): annotation, calibrated_sensor, ego_pose, sample_data and instance fields.
- [Official devkit NuScenes implementation](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py): box_velocity, get_sample_data and keyframe box convention. Numerical comparisons use installed pinned nuscenes-devkit 1.2.0, not an unpinned remote runtime.
- [Official devkit geometry](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/geometry_utils.py): projection and visibility reference.
