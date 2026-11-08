# OCaMo handoff — 2026-11-09

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

## Remote CPU choice — read-only milestone 2026-11-09

S3 is optional, not a prerequisite. The next requirement is an explicitly authorized executable CPU environment. No resource has been started or created.

Fresh authenticated Pod GETs show both existing project Pods EXITED, actions include start, and both still mount the original persistent volume at /workspace in the same data center. Both use runpod/base:1.0.2-ubuntu2404, expose22/tcp, have an SSH public-key setting and empty custom startup arguments. The smaller permissions-check Pod is CPU-only,2vCPU/4GB,10GB container disk, current Pod cost field0.06USD/hour. The preaudit Pod is CPU-only,4vCPU/16GB,20GB container disk, cost field0.16USD/hour. No GPU is requested by either existing configuration. No custom experiment entry command is present in the returned configuration. Historical system-log GETs timed out after bounded reads, so prior boot behavior and current filesystem health remain unverified.

Current CPU catalog GETs (AVAILABILITY,POD,vcpuCount2) report HIGH availability in the volume's data center for cpu3c and cpu3g. cpu3c is0.03USD/vCPU-hour,2GB/vCPU,min2vCPU; cpu3g is0.04USD/vCPU-hour,4GB/vCPU,min2vCPU. Thus a fresh2vCPU/4GB compute-optimized Pod quotes0.06USD/hour, and2vCPU/8GB general-purpose quotes0.08USD/hour. A fresh Pod would need explicit creation, same-volume placement, SSH configuration and boot verification; capacity is time-sensitive and not a guarantee for any particular stopped Pod's original host. Quoted compute rates exclude separate existing persistent-storage charges; no new volume is proposed. No storage-price estimate is invented.

**Preferred authorization proposal:** resume only the existing2vCPU/4GB permissions-check Pod, leaving the larger preaudit Pod stopped. This is the cheapest practical initial configuration for the unchanged per-frame CPU demonstration, subject to measuring actual peak memory and runtime. Same image/volume avoids unnecessary recreation, but stopped container-disk software must not be assumed intact. Build a minimal isolated environment on persistent storage after checking for an existing usable one. Ubuntu24.04 is a reasonable Python3.12 setup candidate, not yet verified runtime compatibility. If that Pod cannot resume or access the volume, report the exact error rather than silently starting another resource. A fresh2vCPU/8GB CPU Pod is the explicit-authorization fallback for more memory/clean placement;4vCPU/16GB costs more and is unnecessary as the first attempt.

For a proposed bounded initial session,2hours of the smaller Pod is0.12USD compute at the quoted rate, plus applicable storage charges. This is a proposal, not spending authorization or an enforced shutdown timer. The execution duration, spend ceiling and stop/cleanup behavior should be explicit in the user's go-ahead. CARLA download remains separately gated.

## Next action after explicit resource authorization

1. Start only the approved Pod; confirm running state, actual SSH connectivity and existing /workspace mount without modifying old experiments.
2. Check existing nuScenes sensor files, checkpoints, CARLA archive/extracted data and persistent CPU environments. Contents remain UNVERIFIED until these checks; local absence/access failure does not establish missing volume data.
3. Only if CARLA is absent, request separate authorization for the author archive to the remote staging destination documented in SETUP.md. No Mac data copy.
4. Reuse or install a small isolated NumPy/SciPy/OpenCV/Matplotlib CPU environment without altering BEVFusion; run unchanged official OCaMo drift first, then LTO, save outcomes and investigate actual failures.
5. Proceed to the clean chronological nuScenes adapter only after official reproduction. No additional S3 setup or audit framework is needed.

Keep raw access responses and local diagnostics in the project's private local reports area; do not publish resource identifiers, credentials, private host paths or internal checkpoint records. A previous detailed audit publication was rejected by automatic approval review for sensitive egress. This handoff deliberately contains only redacted access conclusions, public upstream facts and user-specified logical deployment paths. No experimental implementation or result is claimed.
