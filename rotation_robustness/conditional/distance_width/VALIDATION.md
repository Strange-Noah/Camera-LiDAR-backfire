# Focused Distance × width acceptance

**PASS** for the requested frozen-data, local-CPU descriptive analysis.

| Check | Evidence |
|---|---|
| Accepted identities | Shared loader verifies 53,043 unique sample+annotation keys and exact ordered alignment with accepted geometry, and 53,043×27 hit-array shape. No population intersection or relabeling. |
| Fixed membership | One distance mask and one clean-width mask per cell apply to all 27 conditions. Independent raw-mask recomputation matches every cell count, scene count and SHA-256 denominator identity. |
| Missing/empty/sparse | Total 53,043; 52,664 valid widths and 379 missing. All missing observations retained. Of 20 known-width cells, 17 populated, 14 supported; limited cells flagged. Empty cells are never assigned zero-risk estimates. |
| Frozen numerical rates | Every cell's 27 miss rates independently recomputed from the same frozen hits agrees to 1e−12pp. B, D and G independently derived from these rates also agree to 1e−12pp. |
| B/D/G identity | G=D−B holds in every populated cell to 1e−10pp. |
| Seed aggregation | Mean and sample SD (ddof=1) independently recalculated for D/G at all five σ agree to 1e−12pp. Seeds are not independent GT samples. |
| Marginal reconciliation | Both distance and width marginal counts, all 27 rates, B, D and G reconcile with accepted Phase A summaries to 1e−10pp. |
| Crossings | Observed upward brackets recomputed from mean G, including clean-to-1° only when initial benefit is positive. B≤0 is distinguished from a newly induced transition. No bin tuning or interpolation. |
| Plot reproducibility | All four final PNGs regenerated from the published aggregate JSON are byte-identical in the existing environment. PDF/SVG exports are included; their metadata/IDs are not asserted byte-identical. |
| Visual/units | All four figures inspected. Width colors and distance colors stay consistent within each figure. Axes use rotational σ, raw px, object counts and miss-rate pp. B uses independent panel y scales to show near-group crossings. Missing widths appear in A/D and tables, separately from ordered width curves. |
| Public artifacts | Syntax and relative document links checked; no private input paths, credentials or per-object caches published. Only the additive `distance_width/` directory is part of this change. |

Runtime: existing local Python 3.14.7, NumPy 2.5.4 and Matplotlib 3.11.2, consistent with the retained plotting dependency specification. No new packages were installed. The existing summarizer uses 500 paired scene bootstrap replicates and seed 20261104. G intervals condition on these observed calibration seeds and do not quantify seed-population uncertainty. Seed SD is descriptive. No causal or multiplicity-adjusted trend acceptance is claimed.

The analysis inherits the accepted largest-clean-projected-width selection and frozen widths. It performs no new geometry, inference, E4 matching, official scoring, model development, metadata audit or paid execution. The conclusions are limited by coarse strata, imbalanced support and residual composition. No prior screening or r_rot publication is modified.

Run `validate.py` as documented in [README](README.md) to repeat the focused checks; aggregate-only validation is available without private caches and explicitly reports the omitted raw check.
