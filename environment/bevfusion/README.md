# BEVFusion / TransFusion-L mini inference

Both official detectors completed 8/8 identical mini-val samples on RTX 4090 on
2026-10-20. Checkpoints loaded strictly; no OOM, extension or data-loading errors.
See `docs/bevfusion_deployment_report.md` and `results/deployment/bevfusion/`.
This is inference smoke acceptance, not full nuScenes validation replication.

## Reproducible runtime

Official upstream is pinned in `upstream.json` to
`db75150717a9462cb60241e36ba28d65f6908607`, before the Radar integration.
No upstream source patch is applied. The archived newer HEAD adds Radar to the
shared pipeline, so it is not used for this two-modality baseline.

Accepted runtime: Python 3.8, PyTorch 1.10.1+cu113, torchvision 0.11.2,
MMCV-full 1.4.0, MMDetection 2.20.0, CUDA toolkit 11.3.1, Ubuntu 20.04.
Full package freeze and the built runtime archive hash are recorded under
`results/deployment/bevfusion/`. The Dockerfile is an unbuilt alternative;
the accepted deployment uses the pinned NVIDIA CUDA base image plus a CPU-built
runtime archive. Keep this environment separate from metadata Python 3.12.

## CPU preparation, brief GPU inference, local analysis

1. On Linux x86_64 CPU, `prepare_cpu.sh` builds the environment and custom CUDA
   extensions with FORCE_CUDA, prepares mini infos, validates strict model loading
   and archives `/opt/venvs/bevfusion` plus `/opt/bevfusion`. CUDA toolkit is needed
   for compilation; a GPU is not. Existing artifacts can be reused by hash.
2. `validate_cpu.sh` restores the archive in the minimal deployment image, verifies
   imports, weights, eight real data-pipeline items and the actual runner using
   `smoke.py --preflight-only`. Both processes must exit successfully.
3. Set `BEVFUSION_PHASE=infer_gpu` on a GPU Pod. Use `/bin/bash` as entrypoint and
   `/workspace/cache/bevfusion-runtime/management/environment/bevfusion/bootstrap.sh`
   as its single command. `infer_gpu.sh` restores the archive, imports CUDA ops,
   runs each detector once on eight samples and persists raw predictions. It does
   no model download, package compilation, infos generation, plotting or metrics.
4. Delete the GPU automatically, then analyze the small prediction JSON locally.
   Linux-only dependencies or operations over remote raw data can use a CPU Pod.
   Evaluation, figures, report writing and code development require no paid GPU.

Every new Pod must mount volume `p6vb29fmlu` in EU-RO-1 at `/workspace`. Refresh
stock and price before provisioning. The successful 4090 was $0.89/h. Startup,
archive extraction and runtime imports still occupy billed GPU time; a future
prebuilt accepted image can reduce that overhead. Do not leave GPU Pods idle.

## Shared storage and raw data protection

- `/workspace/checkpoints/bevfusion/`: the two official checkpoints and manifest.
- `/workspace/cache/bevfusion-runtime/runtime.tar.gz`: accepted CPU-built runtime.
- `/workspace/cache/bevfusion-runtime/mini-view/`: symlinks to immutable raw files,
  plus generated train/val infos (323/81 samples).
- `/workspace/cache/bevfusion-runtime/management/`: explicitly staged deployment
  scripts; stage reviewed changes here before creating the next Pod.
- `/workspace/results/deployment/bevfusion-POD/`: logs, state and predictions.

Weights were rehashed after GPU deletion and remained unchanged. Never put model
weights, raw data or runtime archives in Git. Check purchased-volume usage with
`du`; shared-backend `df` is not the 50GB quota. Full trainval does not fit the
current volume and requires separate storage planning.

Raw dataset paths remain read-only. GPU inference runs as `bfbaseline` under
Landlock, writing only to local `/home/bfbaseline/results`. The root supervisor
copies outputs to the shared volume before deleting the Pod. GPU character nodes
and the workload's own `/proc/self/task` write permissions are narrowly allowed
for CUDA initialization and thread names. Do not wrap the final workload in an
extra shell after Landlock; the proc rule must refer to the final Python PID.

## Runner and cache contract

`smoke.py --model fusion|lidar --limit 8 --raw-only --data-view VIEW --output NEWDIR`
uses batch size 1, zero loader workers and six 256x704 views with current LiDAR
plus nine sweeps. Resolved configs record overrides removing unused map labels
and disabling backbone download. Strict loading checks all detector tensors.

Single-rank Gloo supplies upstream world-size queries; inference uses one GPU
without GPU collectives. The rendezvous directory must outlive group destruction.
Reports record tokens, hashes, box frame, classes, timings, allocator peaks and
process peak RSS. `nvidia-smi` memory also includes custom allocations and is much
higher than PyTorch allocator statistics. Raw boxes use keyframe LiDAR coordinates
with bottom-center origin, not nuScenes global export. CPU conversion/evaluation
can follow later. Do not evaluate a partial split against full split ground truth.

Reuse cached LiDAR-only predictions only if tokens, LiDAR points/sweeps/poses,
preprocessing, configs, weights and code identities are unchanged. Cross-modal
alignment changes and direct LiDAR corruption must have distinct identities.

## Independent shutdown

`job_guard.py` runs inside the Pod independently of this chat, token budget or SSH.
It supervises the whole job, copies results and requests Pod DELETE with no GPU
grace delay. Failed jobs are also deleted. Explicit failed acceptance reports
terminate a hung child; 300 seconds without a progress update also triggers cleanup.
This is an inactivity detector, not a fixed total-runtime limit. Longer real jobs
must emit progress updates. API transport failures retry every 15 seconds.
`bootstrap.sh` also has an exit trap for supervisor/startup failures.

`arm_external_guard.py` starts an additional detached Mac watcher. It requires a
live Pod-scoped credential and an awake, connected Mac. Very short successful runs
can finish and revoke that credential before this watcher arms; that happened in
the accepted run, whose Pod-side guard nevertheless deleted the Pod automatically.
An account-level Runpod read returning 404 is the final deletion verification.

Token exhaustion does not stop an already-running Pod-side guard. This is not an
absolute billing guarantee against provider/API outage, revoked credentials or
host failure. The independent guards plus retries reduce that risk; delayed
billing records are not evidence of zero spend. Network-volume storage rent
continues after Pod deletion.

Functional guard tests: `python3 -m unittest discover -s tests -p test_bevfusion_job_guard.py`.
They run real independent processes with a simulated provider API, covering hung
failure, inactivity and retry. The actual successful GPU run separately proves
HTTP 204 deletion followed by account-level 404.
