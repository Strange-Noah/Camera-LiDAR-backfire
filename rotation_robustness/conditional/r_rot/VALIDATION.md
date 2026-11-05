# Focused scientific acceptance

**PASS for the prespecified first-order r_rot analysis on geometrically projectable Car observations.** The exact nonlinear Gaussian/pinhole second moment is not claimed, and 379 geometry-unassigned objects are explicitly retained separately.

| Check | Outcome |
|---|---|
| Injection convention | Original camera-local Gaussian rotation vector and right-multiplied camera extrinsic confirmed from frozen `rotation_protocol.py` |
| Units | Pixel RMS divided by clean horizontal projected width; σ converted from degrees to radians |
| Geometry | Three focused tests pass: numerical Jacobian, on-axis/scale behavior, and visibility rejection |
| Selection | GT-only clean geometric rule; deterministic largest-width camera; no outcome-driven tuning |
| Frozen metadata | Original infos and 13 requested metadata tables verified against accepted manifests |
| Object identity | All 53,043 sample+annotation keys align exactly; 52,664 assigned, 379 unassigned |
| E4 labels | All 27 reused hit-cache hashes equal the accepted extraction ledger; matching never rerun |
| Denominators | Every dynamic σ partition plus unassigned count totals 53,043; each group carries an identity hash |
| Comparable curves | Figures B/C freeze r membership at σ=5°; all five seeds retained separately |
| Statistics | G=D−B checked to 1e−10 pp; five-seed means/sample SD use the existing accepted aggregation |
| Sparse support | Seven empty fixed cells; one 17-object cell flagged; N<200 or scene count<10 omitted from B/C |
| Reproduction | Recomputed full summary is exactly equal; all three PNG figures regenerate byte-identically |
| Preservation | Additive analysis only; no predictions, frozen matrices or scientific manifests regenerated |
| Resources | One CPU-only allocation completed; DELETE 204 and account GET 404 confirmed |

The first allocation request received an explicit backend disk-limit rejection before any resource existed. The disk was corrected only after account reconciliation proved absence. No ambiguous-response replacement was created. Incremental compute/disk cost is conservatively estimated below $0.01 through cleanup confirmation; this is not a final billing statement. Private infrastructure and cost records remain local.

## Interpretation boundaries

The normalization adds some separation at 10–20m and ≥40m, little at 20–30m, and almost none at 30–40m. Strong residual distance association persists within the fixed coarse r bins. Empty/imbalanced cells constrain comparisons. This is partial qualitative agreement with Temporal's distance amplification, not evidence that r_rot fully explains Spatial degradation or that both studies share a universal physical threshold.

Geometric visibility does not model occlusion or network cropping. Some projected corner spans extend outside the raw image. The largest-width camera is one predetermined aggregation rule, not necessarily the camera most responsible for a detection. Small-angle approximation error at 5° is not globally bounded. Broad r intervals can retain different continuous-r distributions across distance groups. Scene-bootstrap intervals inherited from the existing summarizer condition on the observed five calibration seeds. No causal or significance claim is made from the descriptive ordering.

This narrow phase is complete. No alternative normalization search or robustness intervention follows.
