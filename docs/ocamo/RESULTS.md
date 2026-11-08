# First executed OCaMo baseline — 2026-11-09

**Official Python demonstration: PASS for execution/repeatability. Genuine nuScenes pipeline and correction export: PASS for finite, reproducible execution and numerical checks. Reliable nuScenes convergence: NOT ESTABLISHED.** Do not substitute another optimizer or call mixed outcomes a successful general calibration method.

## Official source and environment

Official OCaMo commit `1ac55de3c0eabb02f155cd44f7481117536c9526` remained unchanged (`git diff --exit-code`). A separate Ubuntu24.04/Python3.12.3 CPU environment was created on the existing remote persistent volume. NumPy2.5.4, SciPy1.18.1, OpenCV5.0.0 and Matplotlib3.11.2 imports passed; full exact dependencies are in the lock. The established BEVFusion environment was not altered. No new Pod or GPU was created; only the authorized stopped CPU Pod was resumed.

The author-linked CARLA archive was absent in the bounded project staging/dataset search. It was downloaded directly to remote staging under explicit permission:722112251 compressed bytes,961343167 extracted bytes,200 image/MAT pairs. No sensor/archive/checkpoint/feature data was downloaded to the Mac. Existing remote nuScenes image, keyframe/sweep LiDAR and checkpoints were found readable; this is positive availability evidence, not merely directory registration.

`run_official.py` preserves source, seed42,1,500 cyclic drift observations, original objective, parameters, feature extraction and optimizer. Only plotting/output capture and MAT-read progress are instrumented. Monitoring uses its original200 observations. No hyperparameters were tuned. Times below measure the captured script execution, excluding wrapper startup imports, with some concurrent CPU work; they are not standalone latency benchmarks.

| Demonstration | Observations | Recorded seconds | Mean residual rotation | Final residual | Repeat check |
|---|---:|---:|---:|---:|---|
| OCaMo drift |1500|124.45 /127.43|0.126411° after10-observation burn-in|0.044477°|all trajectory arrays exactly equal|
| LTO drift |1500|151.86 /148.90|0.085186° across all observations; no burn-in|0.049620°|all trajectory arrays exactly equal|

Corresponding uncorrected mean angle is0.586586° for the OCaMo evaluation window and0.583420° for LTO's all-frame window. These are different windows; no head-to-head superiority claim follows. Residuals use SO(3) composition `Exp(theta_est) @ Exp(injected)` rather than component subtraction. The original plots display minus the tracking parameters to compare with injected drift.

The generated OCaMo plot qualitatively follows the author's supplied Python plot: the same seed-driven roll/pitch peaks and yaw trough are followed with the expected sign. Numeric author trajectories, dependency lock and formal tolerance were not supplied. Exact equality with authors' numerical results remains UNVERIFIED; small curve differences are preserved rather than fixed. This is not reproduction of MATLAB or full paper benchmarks, nor an attempt to match SOAC's NeRF calibration metrics.

OCaMo monitoring produced finite outputs: mean clean validity0.976205; mean validity within the injected window0.082781. LTO monitoring was finite:0.996046 clean versus0.031415 injected-window mean. These summarize the official synthetic demonstration, not real-world detection acceptance.

An early capture wrapper incorrectly labeled LTO as having10 burn-in observations; that initial raw summary is retained remotely. The wrapper was corrected to0, and the repeated identical LTO trajectory supplies the corrected aggregate above. No estimator or scientific trajectory was changed. A readiness wait also initially matched its own shell; only that waiting process was terminated before execution. Neither issue was concealed or converted into a calibration failure.

## Genuine nuScenes sequential estimates

The fixed test uses existing v1.0-trainval validation metadata, CAM_FRONT, the first3 sorted scene tokens,2 seeds,0.5° per-axis persistent Gaussian perturbations. All3 scenes have40 genuine paired keyframes, roughly19.4–19.55 seconds acquisition time. State resets per scene/seed. This is2Hz keyframe calibration, not use of the full-frequency sweeps. Ring-aware feature ordering and real geometry are isolated adaptations described in CORRECTIONS.md; SGDSchaul source and the unweighted Gaussian-sum objective are unchanged.

| Fixed scene | Seed | Initial error | Final error | Outcome |
|---|---:|---:|---:|---|
|scene-1|0|0.361783°|0.709224°|worse|
|scene-1|1|0.700111°|0.650713°|improved|
|scene-2|0|0.673620°|0.656078°|improved slightly|
|scene-2|1|0.800284°|0.342394°|improved|
|scene-3|0|0.254208°|2.140673°|drift/worse|
|scene-3|1|0.662098°|2.313563°|drift/worse|

All240 updates are finite;3/6 sequences improve at the final observation. Mean initial error0.575351° becomes1.135441° final. No scene/seed was replaced because of failure, and no early favorable checkpoint was called final convergence. First10 estimates are unchanged burn-in states. The entire six-sequence run was repeated once; theta, matrices and residuals are exactly equal. Recorded first-run processing wall time was19.35 seconds including metadata/feature/geometry work and240 updates, but not the final graph save; this small concurrent run is not a deployment performance benchmark.

The initial loss varies over ±0.25° axis checks. However, known-correct geometry is not always the local minimum: for one first-frame case, clean-oracle loss−612.191 is worse than corrupted−619.030. Thus the objective can be informative but biased/ambiguous for these real features. Feature-order adaptation, visibility, dynamic objects and asynchronous observations are possible limitations; this run does not establish which caused drift. Do not alter the objective to force a positive result.

## Geometry, export and limitations

Actual timestamp-specific world-path and direct LiDAR→camera transforms agree within8.66e−13. Exact inverse-injection calibration recovery error is below4.45e−16. All exported rotations satisfy SO(3), corrected matrices equal corrupted matrices times right correction, and translations remain unchanged. Independent residual recalculation agrees within1e−10°. Corrections are real online states, not oracle substitutes. See CORRECTIONS.md for axes, centering and multiplication conventions.

Clean sensor/projection overlays were generated and retained remotely; they were not downloaded or independently visually reviewed on the Mac. Numeric projection-chain checks pass. This limits visual geometry acceptance and is explicitly not hidden. No BEVFusion inference was run and no mAP/NDS recovery is claimed.

The official and nuScenes convergence line graphs were inspected. [Validation](results/validation.json), [OCaMo plot](results/ocamo-drift.png), [LTO plot](results/lto-drift.png), [nuScenes convergence](results/nuscenes-convergence.png), and small aggregate JSON files are retained. Raw trajectories, actual sensor projections and corrections.jsonl remain under the remote result directory in SETUP.md. Public JSON uses scene aliases; the graph's labels are truncated dataset token identifiers, not private resource/account identifiers.

## Resource closure and acceptance

The authorized CPU Pod was stopped after approximately24.80 minutes and independently read back as EXITED. The other existing Pod remained EXITED; the network volume was retained. Compute upper estimate from start request to stop confirmation is0.02480USD at0.06USD/hour, excluding existing storage and not a final billing statement. No workload or paid resource remains running from this task.

A remote stop guard requiring API credential transfer was rejected by automatic approval review and never executed. A separate local API stop guard was already armed; the Pod was actively stopped and verified well before its two-hour deadline, after which the local guard exited. No credential was exported to the Pod.

**Decision:** execution baseline and correction schema exist and are reproducible; generalized convergence/recovery is not accepted. Stop here, retain the mixed evidence and review it before any new objective, weighting, broader experiment or paid session. No Conditional Weighting or GPU inference is authorized by this result.
