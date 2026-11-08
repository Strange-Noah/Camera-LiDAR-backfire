# OCaMo handoff — 2026-11-09 executed checkpoint

Development branch: noah/ocamo-baseline. The exact checkpoint is the commit containing this document (`git log -1 --format=%H -- docs/ocamo/HANDOFF.md`); the final handoff provides its direct commit link. Foundation:f9acebc5b39076138b656315476a667f339d6109. noah/ocamo-experiments remains reserved at its initial95c3fa452a749a4d88a788413ea8af1ac137a517; no automatic integration or weighting work occurred. Protected historical branches remain unchanged.

## Completed and verified

- Resumed only the authorized existing CPU Pod,2vCPU/4GB, retained original network volume, verified direct SSH and real image/LiDAR/checkpoint file availability.
- Obtained author CARLA data directly on remote storage under explicit authorization. No Mac bulk data copy.
- Built isolated Python3.12.3 CPU venv; installed imports passed and exact lock retained. BEVFusion environment untouched.
- Ran unchanged pinned official OCaMo and LTO drift,1500 observations each, repeated each exactly. Trajectory arrays match exactly. Both monitoring examples,200 observations each, are finite.
- Implemented a separate real nuScenes adapter,3 fixed scenes ×2 seeds ×40 keyframes.240 finite sequential estimates, genuine chronology, reset/burn-in checks and exact repeated estimates. Ring-aware feature ordering and projection are explicit adaptations; original optimizer and unweighted objective formula are preserved.
- Exported real correction records, verified SO(3), composition, unchanged translation, independent residuals and exact inverse oracle. Remote clean overlays exist, but have not been independently visually reviewed on the Mac.
- Stopped the authorized Pod and independently confirmed EXITED; other Pod stayed EXITED; persistent volume retained. Approximately24.80min elapsed, estimated compute0.02480USD excluding storage; not final billing. No new Pod, GPU, or active workload remains.

## Scientific outcome and blockers

Official OCaMo mean post-burn residual0.126411°, LTO all-frame mean0.085186°; signs/curves qualitatively follow authors' examples. Exact author numeric comparison and full paper benchmark reproduction are not established. On nuScenes3/6 final estimates improve and3 worsen, with clear drift on one scene. Mean initial0.575351° becomes1.135441° final. Finite execution is accepted; stable general convergence is not. Do not use a favorable early checkpoint to override final results.

The remaining blockers are scientific reliability and independent visual review of the remote projection overlays, not S3 configuration or an asserted missing dataset. No further optimizer/loss changes, feature redesign, weighting or GPU recovery run was made to hide failures. Future weighting insertion is the individual Gaussian contributions before summation, but remains unimplemented and requires an agreed observable-feature mapping.

## Reproduction and retained artifacts

Read RESULTS.md, SETUP.md and CORRECTIONS.md. Code:ocamo/tools/run_official.py, nuscenes_minimal.py and check_run.py. Config:ocamo/configs/minimal.json. Exact dependencies:ocamo/requirements-cpu.lock.txt. Commands and fresh-output layout are documented in SETUP.md. Small JSON/line graphs are in docs/ocamo/results; raw numerical trajectories, sensor overlays and correction records remain remote at `/workspace/results/ocamo/2026-11-09-first-baseline`. Official source/data/venv stay at `/workspace/staging/ocamo`; existing nuScenes and checkpoints remain in original documented locations. Reuse them rather than download again.

Private local execution/access logs and cleanup evidence stay in the existing local OCaMo session reports, outside Git. No credentials, resource IDs or private host configuration are published. An initially rejected credential-export guard was not executed; the approved local stop guard and explicit early stop achieved resource closure. A wrapper LTO burn-in labeling bug was corrected in the repeat; trajectories were unchanged and original raw evidence retained. All material failures/limitations are in RESULTS.md.

## Next permitted action

Review the mixed convergence and retained remote projection overlays. Any new paid session needs explicit authorization; do not restart automatically. Do not introduce an alternative calibration method, tune this baseline or implement Conditional Weighting without a new agreed scope. No S3 infrastructure is a prerequisite. Preserve existing results and the two OCaMo branches; no task branches or back-merges.

## Latest code-first/directional milestone — 2026-11-09

Continue from this document's commit; reviewed scientific baseline remains immutable d8b86b5df9bd19deb08a726b8000487acbc7ee3e. See CODE_DIRECTIONAL_REVIEW.md and ocamo/diagnostics/directional.py. No estimator/feature/injection change was made. True LiDAR-axis optimizer correction varies slightly under fixed camera-local injection: maximum pairwise0.014290° in the improving scene,0.003746° in the diverging scene (both original seeds measured). This is below the0.057296° finite-difference step and insufficient alone to explain >2° drift. Selected-frame direct matrix/objective checks passed; two existing overlays were privately inspected. Diverging-scene image edges are much denser/noisier; feature correspondence remains an adaptation limitation.

After essential geometry checks, fixed0.5° directions (6 signed axes+8 diagonals), indices0/19/39 in frozen improving scene-2 and diverging scene-3,21 alpha positions produced84 paths/1764 evaluations. Clean endpoint preferred27/42 versus21/42; clean path minimum20/42 versus17/42.26/42 matched direction/frame-position preferences differ across scenes. Directional/scene-dependent objective preference is observed, not a proof of population bias or optimizer-only failure. No weighting conclusion or tuning is warranted. Details, feature counts, plots and all tables are versioned under docs/ocamo/directional. No full-sequence optimization was run.

One authorized30-minute CPU resume session was used; initial400 was followed by one successful retry. Stopped and GET-confirmed EXITED within6.48 minutes; volume retained. Estimated compute upper0.00648USD excluding storage, not final billing. No resources remain running from this task. Raw remote diagnostics are under /workspace/results/ocamo/2026-11-09-directional; old runs unchanged. Public artifacts omit sensor overlays and resource/account identifiers. No new branches or experiment-branch merge.

Next: review feature/crop/noisy-edge mechanisms before proposing modifications; if a focused update diagnosis is requested, use recorded states rather than changing optimizer rules. Fresh paid execution needs authorization. Do not automatically launch feature changes, weighting, a different optimizer or calibration architecture.
