# Progressive two-dimensional findings

Phase A completed and published first (commit `3bc7b8d`). Pair 1 was then saved and published (`3e56c55`) before pair 2 ran. Existing [Distance × LiDAR support](../README.md) and [Distance × r_rot](../r_rot/README.md) outputs are reused as prior evidence, not counted as new discoveries.

## 1. Fixed r_rot × LiDAR support — complete

[Analysis, rationale, figures and table](phase_b/rrot_lidar/FINDINGS.md). Point count continues to separate D strongly inside the two well-supported higher-r groups. At r_rot≥1, D5 is 20.531 / 8.490 / 2.676pp for 0–2 / 3–10 / >10 points. Sparse-support groups also have more initial benefit to lose. Low-r contrasts are absent or sparse, and coarse strata retain composition differences.

This motivated a different next question: does selected-camera direction retain an association after stratifying distance, beyond the geometry/support comparisons?

## 2. Distance × selected camera — complete

[Analysis, rationale, figures and table](phase_b/distance_camera/FINDINGS.md). Direction differences persist inside distance strata: at ≥40 m, front D5=23.409pp versus back 12.727pp. Rankings change across strata and between D and G. At 30–40 m, back-right has higher degradation but lower G than front because it starts with substantially more fusion benefit. Camera labels and frozen perturbation realizations are not isolated causal effects.

## Scientific synthesis and stopping rationale

Distance, GT point support and fixed r_rot are the most useful available diagnostics of degradation. Projected width stratifies risk but shares geometry with distance and r_rot. r_rot provides useful single-variable stratification, but does not absorb point support or all distance dependence. Camera association is only partly accounted for by distance; the two new pairs are limited controls, not a joint adjustment for every confounder.

Initial camera benefit must be tracked separately. Tiny or negative B can produce early positive G with little degradation, including dense-support or near groups. Scene motion proxies mainly distinguish initial benefit or show weak, nonmonotonic degradation differences. They do not establish object-motion effects.

The transferable Temporal lesson is the B/D decomposition and conditional analysis of geometric sensitivity. Rotation sensitivity offers a partial spatial analogue, while residual distance/support/direction effects and unavailable per-object motion prevent a complete transfer of Temporal's mechanistic interpretation. Neither this analysis nor the limited controls establishes a universal explanation.

For future reliability-aware fusion, projection sensitivity and support are candidates for proxy development, while preserving clean benefit as an explicit objective. Current GT point counts, GT projection widths and known injection parameters are research diagnostics; inference-time availability and intervention value have not been validated. No Robust Fusion model development is included.

Stop after these two distinct new pairs and the two reused prior controls. Additional geometry pairs would overlap existing evidence; off-axis and scene-proxy screening is weak; motion, visibility, night and intersection controls require unavailable metadata. No exhaustive search, new extraction, inference or paid execution is warranted in this task.
