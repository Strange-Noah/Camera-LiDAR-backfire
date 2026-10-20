# BEVFusion / TransFusion-L deployment acceptance — 2026-10-20

Both official pretrained detectors passed real single-GPU inference on the same
8 nuScenes mini-val samples. The GPU job exited 0; the Pod-side supervisor issued
DELETE (HTTP 204), and a separate account-level read returned 404. The temporary
CPU validation/results Pod was subsequently deleted and verified 404 as well.

## Models, weights and environment

Official source: https://github.com/mit-han-lab/bevfusion (Apache-2.0, archived).
Pinned commit: `db75150717a9462cb60241e36ba28d65f6908607`.
The newer archived HEAD includes Radar in shared detection preparation; the pinned
pre-Radar revision matches this two-sensor scope. No upstream source patch.

| Model | Config under configs/nuscenes/det/transfusion/secfpn/ | Weight | Bytes |
|---|---|---|---:|
| BEVFusion | camera+lidar/swint_v0p075/convfuser.yaml | bevfusion-det.pth | 163715701 |
| TransFusion-L | lidar/voxelnet_0p075.yaml | lidar-only-det.pth | 33362653 |

Both weights are in `/workspace/checkpoints/bevfusion/` on Network Volume
`p6vb29fmlu`, EU-RO-1, mounted at `/workspace`. Manifest records official Dropbox
URLs, config, code revision and acquired-artifact hashes. Hashes were checked
again after GPU deletion:

- Fusion: `ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122`
- LiDAR: `7f3f1f83754752ea2091543d010a87109d724dc48512a6258c31ef6545619a0e`

These identify the downloaded artifacts; the upstream script does not publish an
independent trusted checksum. No large weights or raw data are committed to Git.

Accepted image: `nvidia/cuda:11.3.1-devel-ubuntu20.04@sha256:e26c89e76b66307f4b7a2bd3e3009c2ec9067be354b12187c44a895657f498f1`.
Python 3.8, torch 1.10.1+cu113, torchvision 0.11.2, mmcv-full 1.4.0,
mmdet 2.20.0. All 12 CUDA extensions built on CPU with CUDA toolkit and FORCE_CUDA.
Actual GPU model forwards subsequently passed. Runtime archive, source and venv
are cached at `/workspace/cache/bevfusion-runtime/runtime.tar.gz`; its SHA256 is
`5d803136dbf26b68baedaf307e55ef55e9956feec590b7b8e940caf78684a3e7`.
Full pip freeze is recorded alongside acceptance artifacts. Dockerfile remains an
unbuilt alternative; the accepted route restores this archive in the pinned image.

## GPU selection and measured smoke results

Successful Pod: `w5ohqdj3jyneoz`, Secure Cloud RTX 4090 24GB, EU-RO-1, $0.89/h.
A40/A6000 were unavailable in this datacenter during selection. The old toolkit
emits sm_86 binaries; Ada compatibility was verified by actual inference rather
than assumed from VRAM. Batch 1, loader workers 0, four CPU compute threads.

| Measurement | BEVFusion | TransFusion-L |
|---|---:|---:|
| Strict checkpoint load | passed | passed |
| Samples completed | 8/8 | 8/8 |
| First measured forward, seconds | 0.128 | 0.095 |
| Mean remaining measured forwards, seconds | 0.105 | 0.080 |
| Runner wall time, seconds | 11.48 | 9.64 |
| PyTorch peak allocated, GiB | 2.26 | 0.47 |
| PyTorch peak reserved, GiB | 3.45 | 0.63 |
| Process peak RSS, GiB | 5.78 | 5.69 |
| OOM / CUDA extension / data errors | none | none |

Two-second nvidia-smi sampling reached **17958 MiB (~17.54 GiB)** total device
memory. This includes custom allocations absent from PyTorch allocator peaks;
use the device measurement for sizing. Sampling can miss a shorter peak.
Forward timings exclude data-loader waiting and whole-Pod startup, and must not
be extrapolated directly into full validation cost. The GPU phase still spent
billed time on container startup, minimal OS prerequisites, archive extraction
and imports. No compilation, model acquisition, infos generation or analysis ran
on this final GPU Pod. A built image can reduce remaining startup overhead.

The Pod was created at 12:02:08 UTC; the guarded job ran from approximately
12:02:43 to 12:03:39 UTC. These timings cover preparation inside the GPU container
as well as both model processes. Deletion was automatic after persistence, with
zero configured grace. The guard log records HTTP 204; account read verified 404.
Billing ledger had not yet posted a row when queried. Do not interpret this as
free use or report an exact finalized cost from the absent row.

## Data and prediction checks

Official mini converter generated infos for 323 train / 81 val samples in a
writable symlink view. Raw dataset is unchanged and Landlock-protected. Pipeline
uses six 256x704 views and current LiDAR plus nine sweeps. Eight timestamp-ordered
val tokens match exactly between models. Both strict state loads were already
validated on CPU (582 fusion / 306 LiDAR tensors).

Predictions contain keyframe LiDAR boxes with bottom-center origin, labels and
confidence, plus resolved config and raw pickle cache. Local numeric checks pass:
finite coordinates, positive dimensions, valid 10-class indices, scores in [0,1].
At score >=0.3, detections per sample were fusion [17,21,30,36,29,32,32,33],
LiDAR [11,25,27,37,29,30,31,34]. Counts are sanity checks, not accuracy evidence.
A sensor/ground-truth overlay and official coordinate export remain CPU analysis
steps before drawing qualitative or accuracy conclusions.

Full output persists at
`/workspace/results/deployment/bevfusion-w5ohqdj3jyneoz/`.
A local copy is under `results/deployment/bevfusion/bevfusion-w5ohqdj3jyneoz/`;
its predictions directory is ignored by Git. Small acceptance and sanity reports
are separately tracked in the parent directory. Mini accuracy was not evaluated;
no official full-val mAP/NDS replication or backfire conclusion is claimed.

## Failures fixed and automatic cleanup

Earlier attempts exposed FUSE ownership differences, missing Python standard
library packaging, CUDA thread-name writes blocked by the sandbox, Git safe-owner
checks and a config serialization API mismatch. Fixes preserve raw-data write
protection and do not change model source. One attempt hung while destroying a
process group and required manual deletion: a heartbeat alone did not detect a
failed but stuck child. That gap was addressed before the accepted run.

The exit hang was traced to removing a FileStore rendezvous directory before
process-group destruction. Keeping it alive through destruction was validated by
both full runner CPU preflights, then both actual GPU processes exiting normally.
The current supervisor also detects failed acceptance reports and 300 seconds
without progress, terminates hung children, persists results and retries DELETE
on API transport errors. Three real detached-process tests with a simulated API
passed: failure-report cleanup, inactivity cleanup, and deletion retry.

The Pod-side supervisor is independent of chat tokens, SSH and the local Mac.
The optional Mac watcher is an extra layer, not the primary shutdown mechanism.
On this short successful run the Pod was already deleted and its scoped key
revoked before the Mac watcher could finish arming. Independent MCP 404 verification
therefore supplied final confirmation. Provider/API outage, revoked credentials
or host failure prevent an absolute billing guarantee; network-volume rent also
continues after Pod deletion. There is no fixed total job runtime limit.

## Next stage and cost discipline

Keep development, manifests, result analysis and plotting on the local CPU.
Use a Linux CPU Pod only for the legacy Linux/CUDA compilation environment or
remote raw-data preparation. Run both CPU preflights before any future GPU start.
Only inference uses GPU, with cached runtime/checkpoints and immediate cleanup.

Before full validation, separately plan full trainval storage (the current 50GB
volume is insufficient), data acquisition, preprocessing and measured end-to-end
throughput on a representative batch. Full-val official references are fusion
mAP/NDS 68.52/71.38 and LiDAR 64.68/69.28; mini smoke success does not establish
those values. Cached LiDAR-only predictions may be reused for camera alignment
experiments only when all LiDAR inputs, poses, transforms and model identities
remain unchanged. No corruption or training was implemented in this phase.
