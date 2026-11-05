# One-dimensional Spatial screening

Phase A is complete before any new 2D selection. All findings are descriptive Car subgroup associations on fixed object memberships.

## Main patterns

Among the eight locally available diagnostics, distance and LiDAR support separate mean degradation most strongly: supported-group D at σ=5° spans 18.549pp across distance bins and 17.245pp across point-count bins. Fixed-reference r_rot and projected width also stratify strongly (12.150 and 10.724pp spans). These ranges are exploratory summaries, not a fair statistical feature-importance ranking: group counts, support and bin choices differ.

- **Distance.** B increases from 0.126 to 8.480pp; D at 5° increases from 0.043 to 18.592pp. Far groups have earlier observed mean crossings. This reproduces the accepted result and motivates geometry/support controls, not a distance-causal claim.
- **GT LiDAR points.** For 0–2, 3–10 and >10 points, B is 9.765, 2.666 and 0.406pp; D is 18.175, 7.466 and 0.930pp. Sparse observations have both a larger initial camera benefit and a larger loss. Dense-group early crossing is small in magnitude and starts from little benefit.
- **Fixed r_rot at 5°.** D rises from 0.194 to 12.344pp across the four reference bins, while B rises from −0.324 to 5.549pp. The lowest bin already lacks clean benefit and must not be called a newly induced crossing. The already published Distance × r_rot analysis shows strong residual distance association; this variable is useful as a diagnostic, not sufficient as a standalone physical explanation.
- **Clean projected width.** D decreases from 12.693pp for <50px to 1.969pp for ≥200px. B also differs (5.519 vs 0.984pp). Width captures appearance scale, orientation, range and selection of the largest-width camera; crossing order is not simply monotonic. The small-width group has 1,051 objects, so seed variation and scene intervals matter.
- **Selected-camera off-axis angle.** D decreases from 7.207 to 5.280pp across the prespecified angular groups, but B and G/crossings do not have a clean monotonic ordering. This is an angle computed from the clean camera-center ray, not measured image displacement or occlusion.
- **Selected camera direction.** D ranges from 4.631pp (back) to 9.444pp (front); G ranges from 1.369pp (front-right) to 5.856pp (front). The camera is chosen deterministically by largest clean projected width, and other cameras still contribute to fusion. Camera-specific frozen perturbations and scene composition can confound this association.
- **Scene-mean ego speed.** D varies only 0.519pp, from 6.638 to 6.119pp across slow to fast scene groups. B varies 1.728pp, so the higher final gap in faster scenes mostly reflects smaller initial benefit rather than larger rotation-induced degradation. This is a scene-level mean, not object-level or frame-level velocity.
- **Scene-maximum ego yaw rate.** D has no consistent monotonic order (6.461, 5.860, 6.588pp); final G differs by only 0.403pp. The full-scene maximum is a coarse proxy and is not Temporal's instantaneous yaw exposure. There is little apparent additional explanatory value in this screen.

## Temporal comparison

The strong camera-benefit/degradation trade-off, distance amplification and sparse-LiDAR sensitivity are qualitatively compatible with Yilia's work. Temporal's mature relative-displacement/ego-motion explanation cannot be claimed replicated: instantaneous motion features are not locally available, and the scene motion proxies show weak Spatial degradation separation. Delay and rotational distribution scale, evaluation populations and model configurations differ.

## Coverage and limitations

All 53,043 eligible Car keys, all 6,019 validation samples and all 27 conditions are reused. Geometric features have an explicit 379-object missing group, not a population intersection. All five seeds remain separate. Scene means/maxima come from the existing accepted metadata-audit scene_support table, uniquely joined to all 150 target scenes. The new continuous bins were documented before inspecting these outcome summaries and were not retuned.

Unavailable locally: physical dimensions, per-object speed, relative lateral/radial velocity, instantaneous ego motion, per-object visibility, day/night and intersection tags. Aggregated metadata histograms cannot safely reconstruct object labels, so they are not substituted. No paid infrastructure was allocated to expand this optional metadata coverage. The conclusions apply to the available-feature screen, not an exhaustive motion/scene investigation.

Every figure has a full-size single-variable version. G is an object GT-miss-rate margin, not official mAP/NDS. Crossings include clean-to-1° when a positive clean benefit is already lost at 1°. B≤0 is labeled separately. Sample SD summarizes the five observed calibration seeds. The existing paired 500-replicate scene bootstrap is retained in JSON and is conditional on these seeds. No multiplicity-adjusted or causal claims are made. A small-benefit flag uses |B|<1pp as a descriptive warning, not a significance criterion.

## Phase B selection

First examine **fixed r_rot × GT LiDAR points**: Phase A shows both strong geometry separation and strong differences in baseline benefit/support. Does point support still separate degradation at comparable geometric sensitivity? Existing Distance × r_rot and Distance × LiDAR results will be reused, not recomputed as new findings. A later pair will be selected only after the first pair is interpreted and published.
