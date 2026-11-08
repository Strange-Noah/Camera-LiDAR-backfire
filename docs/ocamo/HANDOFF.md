# OCaMo handoff — 2026-11-08

## State and recovery decision

Active development branch: `noah/ocamo-baseline`, founded on `f9acebc5b39076138b656315476a667f339d6109`. The checkpoint commit is the commit containing this handoff (resolve with `git log -1 --format=%H -- docs/ocamo/HANDOFF.md`); exact published head is provided in the session handoff. `noah/ocamo-experiments` is reserved from this initial checkpoint. Historical branches remain unchanged.

**BLOCKED before official reproduction. No validated calibration baseline exists yet.** This is an access/source handoff, not a completed deployment or algorithm NO-GO finding. Use a clean execution path starting from pinned official source, not an alternative or patched estimator.

- REUSE: unchanged official source checkout at `1ac55de3c0eabb02f155cd44f7481117536c9526`; source inspection; authenticated read-only control-plane responses; official archive HEAD size.
- REFERENCE ONLY: local bounded metadata/dependency audit; historical BEVFusion environment references and accepted metrics. They do not prove current remote file accessibility or OCaMo behavior.
- DISCARD FROM EXECUTION PATH: the earlier local-first data/runtime proposal and temporary plotting-venv assumption. No estimator/adaptation was implemented, so there is no algorithm pipeline to repair. Preserve private evidence rather than delete it.

## Verified observations and limits

Official inference README, OCaMo/LTO drift, monitoring, SGDSchaul, kernel correlation and feature extraction were read. Upstream working tree is unchanged. Seed42, 1500 drift observations cycling200 frames, 1920×1080 camera, 60° FOV, Gaussian bandwidth9px, KDTree10 neighbors, burn-in10, ±0.001rad finite differences and ±0.0024rad optimizer update bounds are hard-coded. LTO uses27 rotational candidate updates. Original drift is cumulative ±0.0005rad steps, distinct from the future persistent scene/camera injection.

Corner extraction uses adjacent range/azimuth differences and convolution along point order. Do not apply it to arbitrary nuScenes point order without justification. Intrinsics and LiDAR→camera assumptions are duplicated in the objective module, not only the demonstration script. Preserve the faithful official example separately from any nuScenes adaptation. Python Canny differs from MATLAB, as the authors warn; Python demonstration success must not be called full paper reproduction. SOAC remains scientific context, not the implemented algorithm or metric target.

Read-only Runpod GETs succeeded with existing credentials. Existing persistent network volume is registered and both existing project Pods are EXITED with null runtimes. No running compute execution channel was verified. No Pod was created, started, modified or deleted. No Runpod MCP/CLI is currently available in this session; authenticated REST reads were used. The published API specification exposes volume control-plane resources, not a filesystem listing route.

No Runpod S3 credentials were found in the usual environment variables or local AWS credentials file. Thus no current listing/stat/read of `/workspace/datasets/nuscenes`, checkpoints, results, staging, CARLA archive or remote Python environment was possible. **Remote file presence is UNVERIFIED, not missing.** Registration of a volume does not prove its contents, and stopped compute does not imply loss of persistent data. Historical paths are references only.

Current Mac interpreter was Python3.14.7 with NumPy/SciPy/Matplotlib/OpenCV/PyTorch/nuScenes-devkit/pyquaternion unavailable. The prior temporary venv is absent. A bounded local check found metadata but no sensor files at128 checked relative paths; this is strictly local evidence, not an exhaustive disk audit or a remote-volume absence finding. No dependencies, CARLA archive, sensor data or checkpoints were downloaded/installed during this continuation. Official source clone is lightweight source only.

## Priorities and gates

1. Official tracking: NOT RUN; blocked by remote execution/file access and unverified CARLA/runtime availability.
2. Genuine nuScenes adapter/objective: NOT RUN; depends on official reproduction and verified remote sensor observations.
3. Sequential calibration: NOT RUN; no estimates, runtime or convergence evidence.
4. Correction export/oracle: NOT RUN; no detection recovery claim.
5. Conditional Weighting: not implemented; prohibited pending baseline and agreed mapping. Likely future insertion point is per-feature Gaussian contributions before their sum in estimate_kernel_correlation, without implementing weights now.

## Smallest next action and authorization

Provide/configure an existing Runpod S3 read-only access path via the Console's S3 credentials, without posting secrets into chat, or an already-running authorized project execution channel. A regular Runpod API credential is not an S3 credential. Do not create credentials or restart stopped Pods automatically. First list the documented data/checkpoint/staging prefixes and check whether CARLA and a usable CPU runtime already exist.

S3 access can verify storage without launching compute but cannot run OCaMo. If no running environment is available, actual execution may require additional paid compute; that needs a separate explicit proposal/authorization. No new paid compute is currently authorized. CARLA remote download also needs explicit approval after checking for an existing copy. Proposed remote archive destination and minimal dependency setup are in [SETUP.md](SETUP.md).

Keep raw access responses and local diagnostics in the project's private local reports area; do not publish resource identifiers, credentials, private host paths or internal checkpoint records. A previous detailed audit publication was rejected by automatic approval review for sensitive egress. This handoff deliberately contains only redacted access conclusions, public upstream facts and user-specified logical deployment paths. No experimental implementation or result is claimed.
