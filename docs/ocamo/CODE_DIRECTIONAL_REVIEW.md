# Code-first verification and fixed-angle diagnosis — 2026-11-09

Baseline reviewed: `d8b86b5df9bd19deb08a726b8000487acbc7ee3e`. Official source: `1ac55de3c0eabb02f155cd44f7481117536c9526`. Neither estimator, original runs nor feature preprocessing was changed. The new diagnostic loads the exact hashed baseline file and extracts its actual objective function by AST; it does not use a patched calibration pipeline.

**Outcome:** no material coordinate/sign/export ERROR was demonstrated by this focused review. Essential matrix/objective checks passed on the selected observations. The original Gaussian loss formula and SGDSchaul updates are retained, but sensor geometry, centered parameter family and features are explicit ADAPTATIONS, not proof of complete method equivalence. The tested objective often prefers a wrong/intermediate calibration, with direction and scene dependence. This is not evidence that weighting will fix it, or that the optimizer alone caused the earlier drift.

## Equations actually implemented

All transforms use column vectors, T_AB mapping B→A. Let i index an observation, C camera, L LiDAR, E ego and W world. Original code references below are immutable baseline lines.

1. Quaternion storage wxyz is converted to SciPy xyzw; translation fills the homogeneous matrix ([rigid, lines14–15](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/d8b86b5df9bd19deb08a726b8000487acbc7ee3e/ocamo/tools/nuscenes_minimal.py#L14-L15)).
2. `T_WC,i = T_WE,camera(i) T_EC` and `T_WL,i = T_WE,lidar(i) T_EL`, using the two actual sample_data poses/timestamps.
3. `T_CL,i = inverse(T_WC,i) T_WL,i`, `T_LC,i = inverse(T_CL,i) = [Ri,ti]`. Chronology uses sorted camera timestamps within scene, not arbitrary tokens ([lines18–34](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/d8b86b5df9bd19deb08a726b8000487acbc7ee3e/ocamo/tools/nuscenes_minimal.py#L18-L34)). Ego motion is represented; independently moving objects and LiDAR scan deskew are not compensated.
4. The previous Gaussian injection samples one Q per scene/camera/seed, fixed through the scene: `Rc,i=Ri Q`, camera→LiDAR translation ti unchanged ([lines69–75](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/d8b86b5df9bd19deb08a726b8000487acbc7ee3e/ocamo/tools/nuscenes_minimal.py#L69-L75)).
5. The feature coordinate supplied to the objective is `pi_centered=pi_L−ti`.
6. `S(theta)=Exp([theta]×)`, with theta in LiDAR coordinate axes and radians. The estimator retains one state per scene/seed; first ten calls are burn-in.
7. `pi_C(theta)=Rc,iᵀ S(theta)(pi_L−ti)`, followed by Ki projection, positive-depth and actual image bounds, and the retained Gaussian kNN objective ([lines75–84](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/d8b86b5df9bd19deb08a726b8000487acbc7ee3e/ocamo/tools/nuscenes_minimal.py#L75-L84)).
8. `Rest,i=S(theta)ᵀ Rc,i`; `Ai=Rc,iᵀ Rest,i=Rc,iᵀ S(theta)ᵀ Rc,i`; export `T_corrected=T_corrupted diag(Ai,1)` with unchanged translation. Residual is `angle(Riᵀ Rest,i)` ([lines85–88](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/d8b86b5df9bd19deb08a726b8000487acbc7ee3e/ocamo/tools/nuscenes_minimal.py#L85-L88)).

The true optimizer rotation satisfies `S*i=Ri Q Riᵀ`, so `theta*i=Log(Ri Q Riᵀ)`. Thus fixed camera-local Q does **not** imply numerically constant optimizer coordinates under changing acquisition-relative Ri. The original `theta_oracle=Log(Rc Q Rcᵀ)` is algebraically the same because Rc=RiQ. Its sign is consistent with the exported right correction Qᵀ; the optimizer theta is not simply minus the camera-frame injection vector.

### Measured moving-target magnitude

Using all40 stored-scene observation geometries per selected scene and the original two Gaussian seeds:

| Previous scene | Seed | Injected angle | Max target change from first | Max pairwise target change |
|---|---:|---:|---:|---:|
| improving scene-2 |0|0.673620°|0.005925°|0.010493°|
| improving scene-2 |1|0.800284°|0.008266°|0.014290°|
| diverging scene-3 |0|0.254208°|0.001087°|0.001309°|
| diverging scene-3 |1|0.662098°|0.003121°|0.003746°|

The largest pairwise variation is below the0.001rad≈0.057296° finite-difference step and far below the observed >2° final divergence. It is a real model/coordinate ADAPTATION limitation, not strictly zero, but does not alone account for those large errors. There is no claim that it has exactly zero influence on adaptive updates. No injection or existing estimate was redefined.

## Review classification

| Finding | Class | Evidence / remaining limitation |
|---|---|---|
| Matrix direction, quaternion conversion, centering and export sign | ADAPTATION; chosen numerical checks PASS | Actual frozen objective agrees with an independently constructed candidate-matrix objective to <1e−9 loss; projected camera coordinates agree to <1e−9. Oracle endpoints reconstruct clean geometry. No demonstrated sign error. |
| Correction about camera center with fixed C2L translation | ADAPTATION | Official example rotates LiDAR points about its origin before a fixed CARLA transform. Centered features plus variable real projection define a different but stated rotational family. It is not merely unchanged constants or an invariant optimizer reparameterization. |
| Moving true theta | ADAPTATION limitation, measured | Table above; cannot assume constant LiDAR-axis truth merely because camera-local injection is fixed. |
| Gaussian loss and projection validity | Same formula, adapted geometry | `−sum_i sum_j exp(−dij²/(2·9²))`,10 nearest retained image edges, z>0 and image-bound filtering; actual1600×900 replaces1920×1080. No normalization or weights. Original [kernel lines8–19](https://github.com/moravecj/OCaMo/blob/1ac55de3c0eabb02f155cd44f7481117536c9526/python/inference/include/kernel_correlation.py#L8-L19). |
| Finite differences, curvature and state updates | Source retained | Original [SGDSchaul lines29–75](https://github.com/moravecj/OCaMo/blob/1ac55de3c0eabb02f155cd44f7481117536c9526/python/inference/include/SGDSchaul.py#L29-L75): ±0.001rad differences, absolute diagonal second differences, burn-in10, minibatch1, adaptive moments/memory cap5, ±0.0024rad update bounds. No Hessian protection or new update rule. Diagonal adaptation is coordinate-sensitive; equal formula does not imply invariance to a change in parameter axes. |
| Ring labels/order and corner selection | ADAPTATION; physical correspondence not fully verified | Integer ring IDs0…31 checked. Stable per-ring azimuth sort, official get_corners, five-position end/seam exclusion, selected global IDs deduplicated ([baseline lines38–46](https://github.com/Strange-Noah/Camera-LiDAR-backfire/blob/d8b86b5df9bd19deb08a726b8000487acbc7ee3e/ocamo/tools/nuscenes_minimal.py#L38-L46)). Original range jumps/convolution/NMS assume meaningful adjacency ([official lines11–38](https://github.com/moravecj/OCaMo/blob/1ac55de3c0eabb02f155cd44f7481117536c9526/python/inference/include/feature_extraction.py#L11-L38)). No cross-ring convolution, but sorting/gaps, zero-padded convolution/NMS boundary influence, and near-sensor returns remain candidates for feature bias. Five-point trimming is not proof that every boundary influence vanishes. |
| Camera input/Canny | ADAPTATION; pixel identity UNVERIFIED | Official plt.imread JPEG RGB versus OpenCV JPEG decoding and BGR→RGB conversion. Both feed RGB into Canny50/100, not grayscale. Decoder pixel identity was not established. Crop scales from y>360 to y>height/3; image resolution/intrinsics and effective angular bandwidth differ. |
| True edge/3D correspondence | UNVERIFIED beyond the small inspection | Real data and geometry are available, but numeric transform self-consistency and sparse overlays do not establish correspondence for every selected feature. No implementation ERROR is inferred from that uncertainty. |

### Existing overlay inspection and feature distribution

Inspected the frozen first-observation overlays for improving scene-2 and diverging scene-3. There is no apparent gross axis flip or upside-down projection. In scene-2, structural edges are sparse and several red features align with scene structures, while others do not. Scene-3 appears low-light with pervasive noisy Canny texture. This is qualitative evidence, not a controlled illumination experiment. These two small derived diagnostic overlays were inspected privately; no raw sensor files were copied and the overlays are not published.

The lower crop removes image edges above row300 while 3D projected corners there remain in the loss. This behavior is inherited from the original method's edge-only crop, not a newly introduced filtering bug; it can create mismatched candidate support. Across the six frames, total selected corners span1856–2901, but only178–305 project inside the camera. Some selected 3D ranges are extremely small; behind-camera/FOV filtering removes many, but range-jump selection alone does not ensure useful image-edge correspondence. Per-ring counts and radial quantiles are retained in review.json. No filter was changed after inspecting these observations.

## Fixed-magnitude diagnostic and gate

After the above essential numeric checks passed and no invalidating error was demonstrated, evaluate exactly two frozen scenes (previous scene-2 with both final seed estimates improving; scene-3 with both diverging), at zero-based indices0,19,39. These early/middle/late positions were fixed before evaluating diagnostic losses. Use camera-local signed X/Y/Z axes and eight signed cube diagonals, all geodesic angle0.5°, with21 fixed alpha positions0,0.05,…,1. No sequential optimizer, feature retuning or alternative calibration method is run.

For Q=Exp(phi[u]×), `R(alpha)=Ri Exp((1−alpha)phi[u]×)`. The exact optimizer input is `theta(alpha)=alpha·phi·Ri u`, since `S(alpha)=Ri Exp(alpha phi[u]×) Riᵀ`. Then `Rcᵀ S(alpha)(p−ti)=R(alpha)ᵀ(p−ti)`. Every evaluated point-transform and loss was checked against this direct matrix path, and every Q angle was checked to1e−12rad. Using `theta=−alpha phi u` would be incorrect here.

## Results

Lower original loss is preferred. Clean-preferred compares alpha1 against alpha0. A clean endpoint better than the corrupted endpoint need not be the minimum of the path.

| Frozen scene | Frames | Clean preferred over corrupt | Minimum at clean | Minimum at corrupt | Interior minimum |
|---|---|---:|---:|---:|---:|
| improving scene-2 |0,19,39|27/42|20/42|8/42|14/42|
| diverging scene-3 |0,19,39|21/42|17/42|15/42|10/42|

Across both scenes, clean wins48/84 endpoint comparisons; it is the sampled-path minimum37/84. Minima are on this finite1D grid, not global SO(3) minima. Exact argmin/strict comparisons are reported, with no statistical significance or independent-sample interpretation attached to84 correlated paths. The six-frame exploratory sample cannot establish population rates.

Direction/scene examples: −Z prefers clean on all3 improving-scene frames but none of the3 diverging-scene frames. The (−,+,+) diagonal reverses from0/3 to3/3; (+,−,−) reverses from3/3 to0/3. Overall26/42 matched frame-position/direction endpoint preferences differ between scenes. Thus neither perturbation magnitude alone nor an unconditional optimizer-failure explanation describes these cases.

| Scene/frame | Total corners | Image edges after crop | Projected corners across paths | Clean preferred /14 |
|---|---:|---:|---|---:|
|scene-2/0|2665|21121|178–184|8|
|scene-2/19|2630|22045|218–223|8|
|scene-2/39|1856|31029|211–213|11|
|scene-3/0|2579|228018|274–276|7|
|scene-3/19|2434|243236|257–263|7|
|scene-3/39|2901|140641|303–305|7|

Raw sums are not calibrated quality scores across different frames. Projected support changes slightly along paths; clipping/nearest-neighbor changes help explain some discontinuities, but are not isolated as the cause of every preference reversal. The centered plot is only `loss−clean_loss` for display, not a replacement objective or feature-count normalization.

[Improving-scene paths](directional/scene-2-paths.png) · [Diverging-scene paths](directional/scene-3-paths.png) · [Full fixed-path table](directional/paths.csv) · [Endpoint/minimum summaries](directional/summary.json) · [Comparison/count table](directional/comparison.csv) · [Geometry/feature review](directional/review.json) · [Validation](directional/validation.json).

## Recommendation and limits

1. **Implementation inconsistency:** no material matrix/sign/loss mismatch found within tested geometry. Do not patch the accepted baseline based on an assumed sign mistake. The moving-target effect exists but is much smaller than prior large drift.
2. **Directional and scene-dependent objective preference:** directly observed on the fixed paths. It applies to the existing adapted features/objective; it is not proof that original CARLA/paper calibration is biased in the same way.
3. **Optimizer failure:** not isolated. Some individual paths are clean-seeking, but the objective is not uniformly well behaved. Finite-difference/adaptive updates may still contribute; this diagnostic does not exonerate or indict them globally.
4. **Next scope:** inspect feature/crop correspondence, angular-gap and low-light Canny density mechanisms without changing the frozen baseline; compare objective gradients to recorded update directions on a few clean-seeking observations only if specifically requested. Any corrected implementation must be a separate clean path preserving original results, not incremental favorable tuning. No conclusion that weighting will solve this follows.

The authorized existing CPU was stopped and GET-confirmed EXITED within6.48 minutes of the first start request, preserving its volume. Estimated compute upper bound0.00648USD at0.06USD/hour, excluding storage/not final billing. First resume returned400; one subsequent attempt succeeded, with no replacement resource or configuration change. All directional work used existing data/runtime, no download or full-sequence optimization. The local grid checker initially compared floating linspace values exactly; corrected1e−12 equality verifies unchanged values. No baseline scientific files were overwritten.

## Reproduction

Use the existing tested CPU environment/lock from SETUP.md and fresh output directories, after explicit compute authorization:

```sh
P=/workspace/staging/ocamo/venv-cpu/bin/python
U=/workspace/staging/ocamo/official
# REPO is the OCaMo baseline checkout; its nuscenes_minimal.py must match d8b86b5.
$P "$REPO/ocamo/diagnostics/directional.py" --baseline "$REPO" --upstream "$U" --dataroot /workspace/datasets/nuscenes-validation --output fresh-review --phase review
# Inspect review.json and the frozen improving/diverging overlays; stop if invalidating errors appear.
$P "$REPO/ocamo/diagnostics/directional.py" --baseline "$REPO" --upstream "$U" --dataroot /workspace/datasets/nuscenes-validation --output fresh-direction --phase direction
```

The diagnostic source hard-checks the frozen adapter SHA256, uses its exact feature function and actual objective AST, and independently evaluates each candidate matrix with the same h/k/validity convention. It does not invoke SGDSchaul or write the old output directories. The upstream environment/source are the previously verified pinned installation; this focused session did not repeat the full official demonstration or a whole-repository integrity audit. The existing two overlays are at the old run's `nuscenes-3scenes-2seeds/<scene-token>-projection.png`, not its parent directory; an initial lookup in the parent failed and was corrected without rerunning projections. Generated plots/JSON/CSV and all84 summaries remain reproducible from the fixed six sensor observations.
