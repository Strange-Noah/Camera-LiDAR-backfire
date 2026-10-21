# Official validation preparation (CPU only)

Current continuation: the user expanded the existing volume to 90 GB and
authorized CPU acquisition/preflight, a representative GPU pilot, full inference
for both models, CPU evaluation and automatic Pod deletion. Never expand beyond
90 GB. Keep at least 5 GB free. No training, corruption or changes to main/the
teammate branch are authorized. Full baseline results are still NOT_RUN until
actual inference and evaluation complete.

Previous preparation milestone: all ten archive audits passed; official infos cover 6019
samples; strict loading and the original full pipeline passed on 163 complete
real val samples for each model. All 8420 available clean files passed payload
checks; 86102 required files are still missing. Both official evaluator software
tests, a 135-box coordinate round-trip and 10 acceptance tests passed. The CPU
worker and final guard exited 0; deletion was verified through account-level 404.
No GPU was rented at that milestone and the then-50 GB volume was not resized. Full GPU admission stays
BLOCKED_MISSING_PAYLOAD. Full-val model results are NOT_RUN.

This directory is isolated from the previously validated mini/single-frame
deployment. Local branch: `feat/validation-baseline-preparation`. Preserve both
existing draft PRs, main and the teammate's Windows setup branch.

## Files

- `acquire_clean.py`: resume only missing A files from the audited official
  archives, stream without retaining tgz files, atomically publish members,
  verify archive CRC/HTTP byte count, and enforce the 90 GB quota/5 GB reserve.
- `input_protocol.py`: unmodified random-nine-of-ten upstream sweep loader,
  token-derived RNG seeds, actual selected paths and final LiDAR tensor hashes.
  Both CPU and GPU runs must match the same per-token input identity. This is a
  declared single-rank/token-seeded RNG convention, not exact eight-rank RNG parity.
- `test_real_evaluator_cpu.py`: real saved smoke boxes plus legal empty outputs
  for the other official val tokens; software acceptance only.
- `readiness_cpu.py`: full 94522-file, 6019-frame, two-model strict-load,
  input-identity, evaluator, runtime and quota gate before any GPU pilot.
- `infer_validation.py`, `infer_gpu.sh`, `persist_gpu.py`: pilot/full single-GPU
  forward, strict checkpoint load, original complete test transforms, fp16 and
  official cuDNN benchmark flag; atomic per-token persistence and resume.
- `assemble_cpu.py`: verify all raw records, token order and cleanup proof;
  combine/format/evaluate on CPU, then run the acceptance gates.
- `test_persistence.py`: incomplete outputs, checksum failures, idempotent
  replay and overwrite rejection at the durable handoff boundary.

Persistent raw predictions: `/workspace/results/validation-baseline-preparation/clean-predictions/{fusion,lidar}/{sample_token}/`.
CPU merged predictions/metrics: `baseline-{fusion,lidar}/` under the same root.
Run summaries/configs: `clean-predictions/runs/`. GPU bootstrap must use the
independent Pod-side guard, a zero-second completion grace and a 900-second
no-progress watchdog; never use the CPU report-receipt wait on a GPU.
After the CPU task completes, execute `readiness_cpu.py` in the restored CPU
runtime and verify its PASS report before provisioning the pilot. Pilot deletion
must be verified through the account control plane before the full GPU job.

- `audit_manifest.py`: Python 3.12 + ijson 3.3.0. Read original full metadata;
  select official 150 val scenes / 6019 samples from installed devkit splits;
  deduplicate actual pipeline files; stream all ten official archives, checking
  gzip CRC and HTTP length; record actual tar member sizes. No full tgz retained.
  `--acquire-keyframe-lidar` retains only missing current keyframes, capped at
  10 GB per run, to unblock the official converter without full camera payload.
  Directory inventory avoids hundreds of thousands of remote file-stat calls.
- `prepare_maps.py`: acquire official map expansion v1.3, retain only four JSON
  members, preserve original canonical maps. Compressed zip stays on disposable
  CPU disk and is removed. This restores the unused BEV segmentation GT stage
  that was explicitly omitted in the previous smoke pipeline.
- `prepare_cpu.py`: unmodified official metadata records filtered to val-only
  view; pinned official converter with max_sweeps=10; all 6019 token/camera/sweep
  paths checked against the manifest; serialize original resolved configs and
  actual installed official evaluator config. Exercise the official evaluator
  with one valid synthetic box and all 6019 sample keys for both modality declarations. Self-test
  metrics cannot establish model accuracy.
- `verify_payload_cpu.py`: verify every available clean-val sensor file with
  size/hash, full JPEG decode or five-float LiDAR record checks. Missing files
  produce BLOCKED_MISSING_PAYLOAD, never PASS.
- `evaluate_cpu.py`: use pinned upstream `_format_bbox`, including gravity
  center, yaw convention, velocity, ego range filtering, global transforms and
  attributes. Validate trusted raw pickle predictions/token order first, then
  use nuscenes-devkit 1.1.9 DetectionEval on CPU. Both models are supported.
- `baseline-acceptance.py`: hash/provenance, raw validation, exact token coverage,
  finite global boxes, class/size/score checks, process exit, official metrics,
  diagnostic tolerance, and unresolved deviations. NOT_RUN is explicit.
- `test_acceptance.py`: software fixtures cover success and nine rejection/status
  cases. Fixture metrics are not actual model results.
- `preflight_official.py`: same full original transforms for both models,
  including map GT; strict loads and process-group teardown; partial coverage
  cannot produce full GPU admission.
- `progress_relay.py`: relay real non-root pipeline counters to the CPU guard;
  never refresh progress merely because a timer fired.
- `prepare_cpu.sh`, `bootstrap_cpu.sh`, `cpu_handoff.py`: CPU-only orchestration,
  independent guard and verified artifact handoff. The CPU task ends after the
  local controller receives the report packet and creates
  `/root/validation-delivery-ack`; deletion follows with a 30-second grace. The
  guard deletes the CPU Pod after 900 seconds without progress if the controller
  disappears. Never add this receipt wait to the GPU inference job.
- `storage_plan.py`, `finalize_reports_cpu.py`: overlap-aware measured capacity
  calculation and report aggregation; missing payload/iteration blocks admission.

## Persistent locations and resume

Audit output: `/workspace/results/validation-baseline-preparation/`.
Raw canonical inputs remain `/workspace/datasets/nuscenes/` (unchanged).
Derived val view: `/workspace/datasets/nuscenes-validation/`.
Management copy: `/workspace/cache/validation-baseline-preparation/management/`.

All ten archives have completed. Their summaries/indexes and the final manifest
are persisted; the full manifest was independently checked locally against SHA256
`bc68c7dd7a3509354750c30de63ba5c772547e91d09ae852b1bee9bd90fd4982`.
Do not repeat the completed storage audit. With resize authorization received, acquire
only missing clean members on CPU. If an archive audit ever needs recovery,
completed summary + index pairs
skip network transfer. A failed/incomplete archive restarts from its beginning;
already extracted keyframes are inventoried and not duplicated. Keep failed
logs and partial indexes as evidence until superseded.

Copy **all current local validation source** into the isolated management copy
before starting the legacy CPU worker; cached management copies may precede the
final source commit. Git source definitions are authoritative; never assume a
volume cache is the latest code.
Use the exact pinned Ubuntu 20.04 CUDA image + CPU flavor, not the Ubuntu 24.04
audit container: never unpack the OS cache onto a different distribution.
Runtime identity and original scripts live in the parent deployment directory.
No changes to torch/MMCV/CUDA versions or upstream model code are proposed.

Map preparation and existing/new file overlap must be included in a fresh
`du -sx --block-size=1 /workspace` measurement. Keep sensor payload gross bytes
and missing incremental bytes separate to avoid counting existing files twice.
Measured clean payload is 45,914,302,890 bytes (40,601,279,434 missing). B is
71,477,025,411 bytes (65,468,874,208 missing). Fresh existing usage is
26,082,533,888 bytes. With temporary/output budgets and margin, A recommends
90 GB (+$2.80/month); B recommends 110 GB (+$4.20/month), STANDARD $0.07/GB/month.
The user has now expanded the volume to 90 GB and authorized the full baseline
sequence, subject to CPU/pilot gates. No further resize is authorized.

## Execution report contract

Acceptance takes a manifest plus optional per-model execution/evaluation reports.
Execution reports identify `model`, `strict_checkpoint_load`, `exit_code`,
`completed_samples`, `manifest_sha256`, `dataset_version`, `split`,
`unhandled_errors`, `unexplained_deviations`, `checkpoint_sha256`,
`upstream_commit`, `resolved_config_sha256`, `runtime_sha256`, `started_at`,
`finished_at`, and persisted `artifacts` (each `{path, sha256}`). Artifacts must
include `raw_predictions`, `sample_tokens`, `resolved_config`,
`nuscenes_predictions`, and the formatter's `raw_validation` report.

Evaluation reports come from `evaluate_cpu.py` and are linked to the formatted
prediction hash. Official metric references and .01 tolerance are defined in
the acceptance script. Unexplained deviations block reproduction even if
metrics happen to match. The raw-validation report links raw predictions,
token sidecar, infos, and formatted output; it checks the pre-filter raw boxes.

The original smoke runner is not yet a full official-val runner: its BEV GT
omission and worker/cuDNN choices are documented deviations. Do not silently
run that path on 6019 samples and claim the unchanged official pipeline.
The raw formatter round-trip now passed. Full 6019-sample CPU iteration,
chosen worker/distributed flags and full-val execution-report production remain
required before full inference. Full model metrics do not yet exist.
