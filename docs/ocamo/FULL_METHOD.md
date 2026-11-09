# Full-method OCaMo — implementation checkpoint, Gate A pending

**Superseded execution status:** [completed Gate A results](FULL_METHOD_RESULTS.md):
Gate A FAIL, Gate B NOT RUN. The checklist and protocol below preserve the
pre-execution implementation milestone; its pending statements are historical.

2026-11-09. Continue from `f338b966a272a8fd89e3973e5402d5a08947285e`.
This is an implementation and local-test milestone, **not an accepted full-method
reproduction or a new convergence result**. Gate A is UNVERIFIED; Gate B is NOT RUN.
The frozen minimal adapter and its negative results remain unchanged.

## Authoritative references and completeness

Paper: [Moravec & Sara, TRO 2024, accepted manuscript](https://dspace.cvut.cz/entities/publication/1f956869-3ec5-4b0e-9092-456592e5c324), DOI 10.1109/TRO.2023.3347130.
[Supplementary project](https://cmp.felk.cvut.cz/~moravj34/ocamo/) provides tracking videos
and a latency case study, not downloadable preselector parameters.
All source links below pin the original commit `1ac55de3c0eabb02f155cd44f7481117536c9526`.

| Component | Precise reference | New implementation / verification status |
|---|---|---|
| Independent azimuth-ordered scanlines; radial discontinuities, reflectance and wide gaps | Paper III-C1, Appendix A, Table VII; [MATLAB LiDARProcessor](https://github.com/moravecj/OCaMo/blob/1ac55de3c0eabb02f155cd44f7481117536c9526/matlab/include/LiDARProcessor.m) | `core.extract_corners`: separate 32 rings, unique azimuths, circular seam neighborhoods, 11-sample normalized derivative, NMS radii 2/3, thresholds .01/.05, nearer-neighbor rule, two endpoints per .1rad gap. Synthetic geometry checks PASS; actual physical corner/edge inspection pending. Unlike minimal/CARLA, includes reflectance. |
| Camera edges and overlap | Paper III-C1 footnote 2; [MATLAB edge_canny](https://github.com/moravecj/OCaMo/blob/1ac55de3c0eabb02f155cd44f7481117536c9526/matlab/include/edge_canny.m); [official Python caveat](https://github.com/moravecj/OCaMo/blob/1ac55de3c0eabb02f155cd44f7481117536c9526/python/inference/README.md) | Grayscale before Canny; fixed OpenCV 50/100, lower two-thirds for objective. This is an explicit Python-port adaptation, not MATLAB automatic-Canny pixel equivalence. No thresholds chosen on failed scenes. Real-image verification pending. |
| Camera intrinsics / distortion | Paper III-B; [nuScenes official format](https://www.nuscenes.org/nuscenes); [NVIDIA converter conventions](https://nvidia.github.io/ncore/conversions/nuscenes/nuscenes.html) | Use each calibrated_sensor K and actual dimensions. Published nuScenes images are rectified/undistorted; no second undistortion or invented coefficients. Verify runtime image dimensions before use. |
| Motion compensation | Paper III-B Eq 1; [official devkit sweep transforms](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py); NVIDIA converter conventions above | Preserve supplied scan coordinates. Whole-frame transform uses both actual ego poses. nuScenes paper describes supplied motion compensation; NVIDIA explicitly documents compensation to sweep reference time. Fifth float is ring, not per-point time. Do not deskew again or synthesize firing timestamps from azimuth. Exact per-point processing provenance remains a dataset limitation. |
| Physical calibration family | Paper Eq 2; prior CODE_DIRECTIONAL_REVIEW.md | Camera-local right correction A=Exp(theta), C2L estimated R=R_corrupt A, translation unchanged. Candidate points are A^T R_corrupt^T(p-t). True theta=-omega is constant even when acquisition-relative R changes. Independent homogeneous transform checks PASS. This differs explicitly from minimal's LiDAR-axis centered correction; diagonal optimizer adaptation is not coordinate invariant. |
| Kernel objective | Paper Eq 3, III-C2; [Python kernel](https://github.com/moravecj/OCaMo/blob/1ac55de3c0eabb02f155cd44f7481117536c9526/python/inference/include/kernel_correlation.py) | Same negative unnormalized Gaussian kNN sum, sigma9px/k10, positive depth and true image bounds. No weights or new loss. Real-data KDTree equivalence check pending. |
| Stochastic updates | Paper Eqs 5–11, Table VII; [MATLAB SGDSchaul](https://github.com/moravecj/OCaMo/blob/1ac55de3c0eabb02f155cd44f7481117536c9526/matlab/include/SGDSchaul.m) | `PaperTracker`: .001rad central differences, absolute curvature, averaged burn-in accumulation, adaptive moments/memory<=5, proposal limit .0024rad, MATLAB machine eps. First 10 accepted observations do not move theta. Rejections never enter optimizer state. Source-level port and synthetic tests PASS; numerical cross-runtime comparison pending. |
| 42 quality features | Paper III-G1, Fig 7 | 15 five-meter radial bins, each camera-left/right in reference FoV; six image regions with edge counts and mean horizontal/vertical neighborhood anisotropy. Implemented/tested shape42. Region/pixel-boundary and zero-denominator conventions explicitly reconstructed below. |
| Automatic supervision | Paper III-G2 | Three independent [-.05,.05]rad sweeps with .005rad steps; unsuitable when any minimizing displacement exceeds .01rad. Flat/tied profiles conservatively unsuitable. Implemented synthetic labels PASS. Actual labels not yet generated. |
| LogitBoost and scene isolation | Paper V-D, Eq18 | Binary LogitBoost with weighted regression stumps, 100 fixed rounds, five scene folds; independent validation determines threshold for40% data loss. No new dependency. Synthetic learning and leakage guards PASS. Official trained trees/hyperparameters absent from the pinned repository; new training is method-level reproduction, not exact authors' numerical reproduction. |
| Rejected observations / acquisition time | Paper III-C2/G; requested chronology | Rejection freezes every optimizer field, including accepted burn-in count. Log each real timestamp, observed/accepted/update counts. No scene concatenation or cyclic input. Synthetic scheduling PASS; sequence execution pending. |
| Official CARLA regression | Existing RESULTS.md and retained immutable results | Prior exact repeat remains accepted. This new milestone has not rerun CARLA; one unchanged official regression is included in the proposed bounded CPU execution. |

The published MATLAB likelihood divides by projected-corner count and image-edge
count, whereas paper Eq3 and the Python example use the unnormalized sum. The new
path deliberately follows Eq3/Python for the loss and the MATLAB implementation for
the update/burn-in mechanics. These source differences must not be described as
complete MATLAB/Python numerical equivalence. The MATLAB drift example disables
absolute monitoring bounds; this tracking path does likewise. Waymo-trained
monitoring variances/bounds are not imported as nuScenes tracking truth.

## Fixed reconstruction choices and limitations

No official frame-preselection model, feature extractor, label generator or
training pipeline appears anywhere in the pinned public tree. Its intensity
branch is also commented out for CARLA. The new code reconstructs the paper's
method without claiming unpublished assets were obtained.

- Scanline circular padding supplies real samples at the arbitrary -pi/pi seam;
  missing azimuth intervals remain collapsed as in the paper. Returned source IDs
  allow actual feature inspection. Synthetic flat scans produce no seam corners;
  range steps select the nearer surface, reflectance steps and gap endpoints are
  present, and rings never share convolution neighborhoods.
- Six image regions are reconstructed as a 2x3 grid over the bottom three-quarters.
  Neighbor counts use offsets +/-1..10 excluding center; denominator is floored at
  one for finite anisotropy. Exact unpublished edge/boundary conventions and
  original classifier settings are not established. These are fixed before new
  data evaluation, recorded as adaptations, and cannot be tuned on test scenes.
- Training uses reference clean calibration to obtain automatic labels. At
  evaluation, FoV quality features use the available corrupted nominal reference,
  never the true residual or an oracle correction. This introduces an explicit
  train/deployment FoV shift that must be reported, not hidden by oracle selection.
- Known prior failed scenes remain in the comparison. An additional predetermined
  held-out set, never used for classifier training or threshold choice, prevents
  the original six trajectories from becoming the only evidence.
- MATLAB automatic Canny and Python OpenCV Canny are not identical. If the
  documented Python adaptation prevents reliable convergence, Gate A remains
  closed; no favorable optimizer/edge tuning or high-sigma launch follows.

## Predetermined Gate A execution

Scene tokens are sorted lexicographically within the existing validation metadata
view, before computing new losses. Zero-based ranks:

| Population | Scene ranks | Observation policy |
|---|---|---|
| Training |3..14|8 evenly spaced real keyframes per scene|
| Independent threshold validation |15..18|8 evenly spaced real keyframes per scene|
| Previous evaluation |0..2|All available unique paired observations|
| Additional held-out evaluation |19..21|All available unique paired observations|

Two seeds0/1 and sigma .5 degrees per axis exactly reuse the old SHA-derived
standard-normal vectors. Initial SO(3) angles are reported separately from sigma.
No missing scene is replaced according to loss outcomes. Insufficient scenes or
one-class training causes an explicit stop, not a fake successful classifier.

Compare frozen minimal, reconstructed full/no-preselection, and reconstructed
full/preselection on the **same** observation pairs and perturbations. Keyframe
and genuine higher-frequency pairs are reported separately. Each camera and LiDAR
record is used at most once within a cadence; nearest pairing is limited to60ms.
Missing files and unpaired observations are counted. Higher-frequency samples
must physically exist on the current volume; metadata alone is not availability.
If the higher-frequency set equals keyframes, report unavailable and do not rerun
the same set under a new cadence label. More observations do not extend the
roughly20-second physical scene duration.

All outputs go to a fresh remote directory. Record initial/final error, full
trajectories, accepted/rejected counts, valid updates, post-burn-in mean and actual
acquisition seconds. Prefix summaries reuse recorded trajectories. Invalid
curvature is reported separately and does not count as a valid update; no Hessian
damping is introduced to turn invalid arithmetic into a successful run.

Before Gate A can pass, inspect actual features/overlays and compare objective and
optimizer numerics to the cited sources. Then use this predeclared exploratory
convergence gate: preselection method improves at least9 of12 final scene/seed
errors, median final/initial ratio<=.75, no final error>max(2*initial,2 degrees), and
at least20 valid post-burn updates per evaluated trajectory. This is a project
acceptance rule, not a paper theorem or statistical confidence claim. Report all
12 outcomes regardless. Fewer valid observations means insufficient evidence;
systematic growth over observed time is reported as drift even on short scenes.
Material geometry/feature failures override aggregate thresholds.

## Current evidence and next permission

Eight focused tests PASS locally with an existing bundled NumPy runtime; all new
Python files parse. No packages installed. SciPy/OpenCV/image execution and real
sensor checks require the already existing isolated remote CPU environment.
Gate A real results and new figures do not exist yet. No new/resumed paid work,
GPU work, sensor download, CARLA download or Gate B execution has occurred.

The existing2vCPU/4GB CPU Pod was read as EXITED with its original /workspace
network-volume mount and rate0.06USD/hour. Proposed bound:90 minutes, compute
ceiling0.09USD excluding existing storage, stop immediately on completion/failure
and independently verify EXITED. Reuse datasets, official source, CARLA and venv;
no new packages are required by the implementation. Authorization is required
before resuming this paid resource.

## Gate B remains closed

The frozen spatial result inputs contain sigma0,1,2,3,4,5 degrees (five seeds for
nonzero values), unlike the proposed .5,1,2,4 grid. If Gate A is later accepted,
start a small paired .5,1,2 progression; preserve3 and4 for historical consistency,
and only consider5 if earlier cases justify it. This is a deferred plan, not an
authorized run. No high-sigma executor or Conditional Weighting is implemented.

Implementation and commands: [full_method README](../../ocamo/full_method/README.md).
Exact implementation checkpoint: `git log -1 --format=%H -- ocamo/full_method`.
No new convergence result commit is claimed until actual execution.
