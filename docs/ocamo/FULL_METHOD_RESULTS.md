# Gate A executed — convergence acceptance failed; Gate B not run

2026-11-09. Actual execution source:
[`1e6b7a1915ed5c2b47bc40aeff41c205ba6f8ea9`](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/1e6b7a1915ed5c2b47bc40aeff41c205ba6f8ea9),
following implementation checkpoint
[`d75ac5855bef43d5fa8fa1bb3637c01ed99c0f4b`](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/d75ac5855bef43d5fa8fa1bb3637c01ed99c0f4b).
See [paper-to-code checklist](FULL_METHOD.md) for the reconstructed components and
explicit MATLAB/Python/unpublished-asset limitations. This is a completed, bounded
Gate A comparison, **not an accepted reliable full-paper baseline**.

## Results on the fixed population

Six predetermined scenes, seeds0/1, sigma0.5 degrees per axis, identical persistent
camera-local rotations across all three methods. Initial geodesic angles actually
range0.254208–1.551537 degrees; mean0.692569 degrees. Sigma is not the realized
rotation angle. The three previous scenes remain present alongside three additional
held-out scenes. No final scene entered training or threshold validation.

| Method | Genuine observations per trajectory | Final improved | Final degraded | No valid update | Mean initial → final error | Median final/initial |
|---|---:|---:|---:|---:|---|---:|
| Frozen minimal conventions |40–41|6/12|6/12|0/12|0.692569° →0.832587°|0.989302|
| Reconstructed full / no preselection |40–41|9/12|3/12|0/12|0.692569° →0.516814°|0.763218|
| Reconstructed full / preselection |40–41|8/12|2/12|2/12|0.692569° →0.540137°|0.830451|

Each method sees the same241 physical observations ×2 seeds =482 records;
all three together produce1446 records. These correlated trajectories are not12
independent population-level trials. “Improved” requires a valid update and a
reduction greater than1e-8 degrees, a numerical comparison tolerance only.

| Method | Accepted / rejected records | Valid updates | Trajectories with >=20 valid updates | Post-burn means available |
|---|---|---:|---:|---:|
| Minimal |482 /0|362|12/12|12/12|
| Full / no preselection |482 /0|362|12/12|12/12|
| Full / preselection |300 /182|189|4/12|10/12|

No INVALID_CURVATURE or nonfinite arithmetic occurred. The other351 accepted
records were burn-in;182 were rejected. Rejected frames did not advance optimizer
state or burn-in. Each trajectory has its own initial and final SO(3) error,
accepted/rejected counts, valid updates, actual seconds and post-burn mean in
[summary.csv](full_method/executed/summary.csv). Post-burn means include accepted
observations after accepted-count10; their coverage differs across methods and
must not be compared as though all observations/populations were identical.

### Preselection outcomes, without omitting failures

| Scene prefix | Seed | Initial | Final | Accepted / valid updates | Outcome |
|---|---:|---:|---:|---|---|
|01452fbf|0|0.361783°|0.763886°|19 /9|Degraded|
|01452fbf|1|0.700111°|0.572530°|18 /8|Improved|
|01e4fcbe|0|0.673620°|0.397029°|34 /24|Improved|
|01e4fcbe|1|0.800284°|0.378877°|34 /24|Improved|
|034dee16|0|0.254208°|0.254208°|5 /0|No calibration update|
|034dee16|1|0.662098°|0.662098°|6 /0|No calibration update|
|265f002f|0|0.775881°|0.374638°|28 /18|Improved|
|265f002f|1|1.551537°|0.684259°|27 /17|Improved|
|26a6b03c|0|0.987283°|0.832410°|26 /16|Improved|
|26a6b03c|1|0.379145°|0.577999°|27 /17|Degraded|
|2abb3f35|0|0.590664°|0.470059°|38 /28|Improved|
|2abb3f35|1|0.574210°|0.513653°|38 /28|Improved|

Previous/held-out improvement counts respectively: minimal3/6 and3/6; full
without preselection4/6 and5/6; full with preselection3/6 and5/6. These are final
outcomes, not favorable intermediate checkpoints. Some trajectories oscillate or
lose earlier gains; more elapsed time cannot simply be assumed to repair them.

## Frame-preselection training evidence

Training:96 automatically labeled observations in12 scenes,29 suitable/67
unsuitable. Validation:32 observations in4 separate scenes,15 suitable/17
unsuitable. Five-fold cross-validation splits whole training scenes. The100-round
LogitBoost regression-stump ensemble and42 features were fixed before evaluation.
Threshold -2.330929 was chosen on independent validation scores for40% requested
data loss; observed validation rejection was13/32=40.625%. Evaluation data loss
was182/482=37.759%, not an enforced quota on test scenes.

The classifier generalizes poorly in this modest reconstruction:

- Cross-validation: TP6/TN46/FP21/FN23, accuracy52/96=54.167%, below the constant
  majority-class reference67/96=69.792% on these labels.
- Independent validation at the chosen threshold: TP8/TN6/FP11/FN7,
  accuracy14/32=43.75%, suitable precision42.105%, recall53.333%; majority-class
  reference17/32=53.125%.

Consequently, this does **not** establish that frame preselection improves the
nuScenes baseline. Its average final error is higher than the full no-preselection
variant. Holding the low-light scene avoids the minimal pipeline's drift there,
but a held estimate is not successful recalibration. No training round count,
threshold, feature convention or scene selection was changed after seeing this.
Original trained parameters and unpublished ensemble settings remain unavailable;
these are method-level reconstructed assets, not an exact original model.

[Trained model and folds](full_method/executed/preselection-model.json) ·
[Automatic labels and42-D features](full_method/executed/training-labels.json).

## Geometry, features and original-baseline preservation

- Existing CPU imports and8 targeted runtime tests PASS. The separately added
  reporting-edge-case test brings current local tests to9 PASS; it was not used
  to claim an extra remote calibration run.
- The unchanged official CARLA example ran1500 observations. Every saved array
  matches the accepted original exactly, trajectory SHA256
  `0e494d11738a72771e844714897fa6170b2a44c1e5ee2afe313fd0ee2373513f`.
  Mean post-burn residual0.126410705786°, final0.044476999756°.
- The old minimal replay matched240 original accepted observations with maximum
  residual difference1.85731e-11 degrees. The frozen file hash was verified;
  no accepted minimal implementation or output was modified.
- Six fixed real frames from the previously improving/diverging scenes passed
  oracle/homogeneous matrix and independent brute-force kNN loss comparisons
  (loss tolerance1e-8). Their6 private projection overlays were inspected.
- Range/intensity/gap features include recognizable vehicle, curb and pole
  structures, but also numerous foliage/texture/noise-related or noncorresponding
  features. Selected pre-FoV radial quantiles include near-origin returns. Only
  276–580 of4574–5608 selected corners project inside the image on these frames;
  grayscale lower-crop edge counts span19090–170157. Clean median nearest-edge
  distances span1.94–36.17px. Dense noisy edges can give short distances without
  establishing physical correspondence. These checks verify geometric consistency
  and limited real structural support, not that every selected feature is correct.
- The new whole-frame geometry respects both acquisition-time ego poses. Existing
  sweep coordinates are not deskewed again; no firing timestamps were invented.
  Exact per-point compensation provenance remains a documented dataset limitation.

The new/full comparison bundles reflectance extraction, grayscale preprocessing,
camera-local parameter axes and MATLAB-style averaged burn-in. It is **not** a
factorial attribution of the improvement to any one component. It does not prove
complete MATLAB numerical equivalence or invalidate the original negative runs.

[Official regression](full_method/executed/official-regression.json) ·
[Replay validation](full_method/executed/replay-validation.json) ·
[Feature numerical evidence](full_method/executed/feature-verification.json) ·
[Actual execution source hashes](full_method/executed/executed-source.json).
The feature-check file preserves its original PENDING visual-review markers;
the inspection finding above is recorded separately here rather than overwriting
that raw execution evidence. Sensor overlays remain private.

## Acquisition horizon and missing higher-frequency inputs

The fixed22-scene inventory found all required keyframe pairs. The six evaluation
scenes contain40–41 observations spanning19.4–19.75 seconds. LiDAR sweeps exist,
but higher-frequency camera JPEGs are unavailable in the inspected existing roots.
For example, the fixed first scene has190 non-keyframe CAM_FRONT records and zero
corresponding files in each of the validation, original-nuScenes and smoke roots.
This is a verified limitation of the inspected data locations, not proof that no
copy exists anywhere else on the persistent volume.

No large download was authorized or performed. Higher-frequency runs were marked
UNAVAILABLE_NO_ADDITIONAL_OBSERVATIONS. No repeated keyframes were called new data,
and no scenes were concatenated. Prefixes10/20/40/(41 when present) reuse genuine
trajectories. There is no evidence for80/160-observation prefixes or a physically
longer sequence. Short horizons restrict accepted updates, especially after
preselection, but do not explain away observed increases/oscillations in residual.

[Availability counts and fixed populations](full_method/executed/availability.json) ·
[Complete numerical trajectories](full_method/executed/trajectories.csv).

## Figures

[Residual versus acquisition seconds](full_method/executed/keyframes-elapsed_seconds.png) ·
[Residual versus accepted observations](full_method/executed/keyframes-accepted_count.png) ·
[Residual versus valid updates](full_method/executed/keyframes-valid_updates.png).
All12 scene/seed outcomes and all methods are shown. Zero-update trajectories
collapse at x=0 in the update-count figure; use the physical-time plot and table
to distinguish a held estimate from recovery. Axes and per-scene ranges differ;
do not compare apparent slopes solely by panel height.

## Acceptance and smallest next correction

The predeclared Gate A checks remain unchanged:

| Check | Observed | Status |
|---|---|---|
|>=9/12 meaningful final improvements with preselection|8/12|FAIL|
|Median final/initial<=.75|.830451|FAIL|
|No final>max(2*initial,2 degrees)|0 such failures|PASS|
|All12 trajectories have>=20 valid updates|4/12|FAIL|

**Gate A FAIL; Gate B NOT RUN.** Numeric geometry/objective checks passed, but
preselector generalization, coverage and final convergence are not reliable enough
for interpreting a high-sigma robustness experiment. The sample includes wet/rainy
and low-light scenes outside the paper's main daylight/non-heavy-rain population;
this is a practical nuScenes comparison, not a population-level refutation of OCaMo.

The smallest scientifically meaningful next correction is to validate the
reconstructed42-D feature conventions and LogitBoost learner against the original
method on the independent training/validation populations. Its below-majority
validation performance is a concrete blocker before changing calibration weights
or optimizer rules. More existing training observations and exact original assets,
if obtainable, are preferable to selecting successful test scenes. Address the
missing non-keyframe camera inputs separately before claiming a longer-cadence
convergence result. Do not automatically restart compute or download those inputs.

One reporting issue was found: raw strict `<` counted two held-state differences
of -5.55e-17 and -1.11e-16 degrees as improvements. The immutable remote
[raw-summary.json](full_method/executed/raw-summary.json) is preserved. The report
and CSV classify these NO_VALID_UPDATE; future summary code now requires valid
updates and1e-8-degree comparison tolerance. No calibration values changed and no
experiment was rerun to fix this reporting artifact.

The authorized original CPU was stopped and independently GET-confirmed EXITED,
retaining its volume. Elapsed upper bound795.35 seconds (13.26 minutes); estimated
compute upper0.01326USD at the read0.06USD/hour rate, excluding existing storage
and not a final invoice. The initial platform approval timeout executed nothing;
one permitted retry resumed the same Pod. No replacement resource, GPU work,
dependency install, new sensor download or Gate B workload occurred.

[Machine-readable final acceptance](full_method/executed/acceptance.json).
Implementation and original artifacts remain on noah/ocamo-baseline; noah/ocamo-experiments,
main, historical branches and teammate branches are unchanged. The result/report
commit is the commit containing this file (`git log -1 --format=%H --
docs/ocamo/FULL_METHOD_RESULTS.md`); the final handoff links that exact commit.
