# Spatial conditional screening

## Phase A

Eight locally available Car diagnostics were screened before new pair selection: distance, fixed r_rot at 5°, clean projected width, selected-camera off-axis angle, GT LiDAR points, selected camera direction, scene-mean ego speed and scene-maximum ego yaw rate.

[One-dimensional findings](phase_a/ONE_DIMENSIONAL_FINDINGS.md) · [Compact screening table](phase_a/screening_compact.csv) · [All seed values](phase_a/screening.csv) · [Full statistics and paired scene intervals](phase_a/summary.json) · [Figures](phase_a/figures)

Distance and LiDAR support separate degradation most strongly in this descriptive screen. r_rot and projected width also stratify risk. Scene motion proxies mostly separate initial camera benefit or show little degradation ordering. An early crossover alone is not evidence of a large misalignment effect.

## Phase B and synthesis

Two new pairs were completed sequentially: [fixed r_rot × LiDAR support](phase_b/rrot_lidar/FINDINGS.md), then [Distance × selected camera](phase_b/distance_camera/FINDINGS.md). [Cumulative findings and stopping decision](TWO_DIMENSIONAL_FINDINGS.md) combine these with the two reused prior controls.

Point support remains informative inside high-r groups; direction differences remain within distance strata but lack a universal ranking. These descriptive controls leave residual composition effects. GT diagnostics suggest future reliability proxies, whose inference-time availability and intervention benefit remain untested. This closes the exploratory phase without beginning model development.

## Method and scope

Reuse the accepted 6,019-sample validation cohort, 53,043 Car annotation identities and 27 original hit/miss conditions. No inference, matching or official scoring is repeated. All groups remain fixed across σ and seeds. Three hundred seventy-nine observations lacking the prior clean projection are an explicit missing group for geometric diagnostics.

The per-group B/D/G summaries and 500-replicate paired complete-scene bootstrap use the existing `core.py`; scene intervals condition on the five observed calibration seeds. All five seed values and sample SD are retained. A group is flagged for limited support if N<200 or fewer than 10 scenes contribute. |B|<1pp flags a small clean benefit; B≤0 means no initial clean advantage. Observed mean crossings include the clean-to-1° bracket when appropriate, with no claim of a calibrated physical threshold. Exploratory ranges and trends are not multiplicity-adjusted or causal.

`bins.json` records the accepted distance/r/point bins and simple new bins chosen before inspecting outcome summaries. No retuning was performed. Off-axis angle uses the clean selected-camera center ray; camera selection is the prior largest-clean-projected-width rule. Neither is the realized displacement or an isolated camera contribution. Prior r_rot is a linearized distribution-derived sensitivity feature, not seed-specific rotation.

Scene features are full-scene means/maxima from the existing metadata-audit `scene_support.csv`, uniquely joined to all 150 target scenes. They are not instantaneous ego speed/yaw. Physical dimensions, per-object and relative motion, visibility, day/night and intersection labels were unavailable as safely joinable local per-object data and were not reconstructed from aggregates. This is broad screening of available local features, not an exhaustive metadata study.

## Reproduction

Use the original accepted object/hit cache and r_rot geometry cache. The scene table is retained in this repository at `results/metadata_audit/independence/scene_support.csv`. NumPy/Matplotlib versions follow the existing plotting environment specification.

```sh
python screen.py --labels ACCEPTED_EXTRACTION --geometry ACCEPTED_GEOMETRY.json.gz --scenes SCENE_SUPPORT.csv --output fresh-phase-a
python screen.py --regenerate phase_a/summary.json --output regenerated-figures
# Run a selected pair only after Phase A findings are saved; use a fresh output directory.
python screen.py --labels ACCEPTED_EXTRACTION --geometry ACCEPTED_GEOMETRY.json.gz --scenes SCENE_SUPPORT.csv --phase-a phase_a --pair r_rot5,lidar_points --output fresh-rrot-lidar
python screen.py --labels ACCEPTED_EXTRACTION --geometry ACCEPTED_GEOMETRY.json.gz --scenes SCENE_SUPPORT.csv --phase-a phase_a --pair distance,camera --output fresh-distance-camera
python screen.py --regenerate phase_b/distance_camera/summary.json --output regenerated-camera-figures
```

`screen.py` only groups existing labels. Per-object data and private audit records are not distributed. This phase uses local CPU only. No model training or robustness intervention is part of the task.

[Focused validation](VALIDATION.md). Existing [Distance × r_rot](../r_rot/README.md) and [Distance × LiDAR points](../README.md) remain accepted prior analyses and will not be presented as new pair discoveries.
