# Geometric conditional comparison

[Seven scientific answers](GEOMETRIC_CONDITIONAL_COMPARISON.md) · [Focused acceptance](VALIDATION.md) · [Fixed protocol](protocol.json)

This is an additive comparison using unchanged accepted caches. Physical metadata was obtained from the retained official [trainval metadata source](https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz) and read locally, avoiding paid compute. Only six GT/pose JSON tables were needed; no images, point clouds or projections were downloaded/generated. Private object-level caches stay local.

## Reproduction

Use the existing NumPy/Matplotlib plotting environment specified by `../../evaluation/plot-requirements.txt`; the physical extractor requires only Python's standard library. From this directory, with fresh output directories:

```sh
# METADATA is the v1.0-trainval table directory, with sample_annotation, sample,
# sample_data, ego_pose, calibrated_sensor and sensor JSON files.
python extract_physical.py --objects ACCEPTED_EXTRACTION/objects.json --metadata METADATA --output fresh-physical
python compare.py --labels ACCEPTED_EXTRACTION --geometry ACCEPTED_GEOMETRY.json.gz --scenes SCENE_SUPPORT.csv --physical fresh-physical/physical.json.gz --output fresh-statistics
python plots.py --results results --output fresh-figures
python validate.py --labels ACCEPTED_EXTRACTION --geometry ACCEPTED_GEOMETRY.json.gz --scenes SCENE_SUPPORT.csv --physical fresh-physical/physical.json.gz --summary results/summary.json
```

`SCENE_SUPPORT.csv` is the existing repository-root `results/metadata_audit/independence/scene_support.csv`, read by the inherited loader; no new motion analysis is performed. The extractor joins annotations by the exact accepted token, checks sample/instance/scene/translation and accepted distance, then computes a single orientation-aware transverse physical span. It never changes eligibility or matches predictions. `source_hashes.json` records table hashes; published provenance also records the metadata archive hash. Recompressing JSON may alter gzip header bytes; geometric values and identities, not gzip bytes, are the scientific cache identity.

The common valid mask is fixed by the existing pixel-width/r caches (52,664). Physical q exists for all 53,043 objects, including the 379 separately reported projection-unassigned records. All indicators use the same common set in inferential and held-out comparisons. Public tables contain aggregate counts/statistics and denominator hashes, not raw annotations.

`compare.py` retains both 500-replicate scene-only and scene-plus-seed uncertainty, all original seed values, complete D/G curves, B, scene coverage, missing-population summaries and crossover categories. Empty resampled denominators are undefined; percentile intervals and contrasts omit nonfinite draws and report applicable replicate counts. Sparse cells are retained descriptively, not used by supported trend/contrast tests. Bootstrap draws are paired across cells, variables and σ; seed indices are separately resampled per σ and shared across cells. Five-seed SD is sample SD, not a GT sampling interval.

The primary curve statistic is the arithmetic average across σ1…5. `contrasts.csv` includes B, D/G at every σ and curve averages; endpoint and adjacent comparisons use the supported bins in the fixed hypothesized risk order. Endpoint labels are printed explicitly because available support differs by stratum. `trends.csv` includes strict D ordering and supplemental strict/non-strict first-sustained-crossover ordering. Bootstrap shares use finite eligible draws, with eligibility reported for crossing tests. These are descriptive frequencies, not multiplicity-adjusted significance tests. Earlier crossing is never credited when clean fusion already lacks advantage.

OLS designs in the scene-held-out check contain fixed distance-bin main effects plus fixed indicator-bin main effects. All five target D curves are fitted simultaneously without regularization; no high-capacity model or hyperparameter search is used. A deterministic permutation of sorted scene IDs, then index modulo five, assigns scene folds. Coefficients, fold counts, rank and a fold-map hash are retained. Only aggregate fold records are published. Paired loss intervals condition on this one split scheme and fixed fitted predictions; there is no refit inside the loss bootstrap or claim of repeated-training uncertainty. It tests D prediction, not crossover prediction. Pairwise indicator contrasts reuse the same OOF errors without fitting more models.

## Output map

- [Full cell statistics, intervals and provenance](results/summary.json)
- [Compact cross-variable comparison](results/cross_variable_comparison.csv)
- [All cell rates/seeds](results/cells.csv)
- [Conditional D/G/B contrasts](results/contrasts.csv) · [Trend/order support](results/trends.csv)
- [Cell crossover distributions](results/cell_crossovers.csv) · [Paired crossover comparisons](results/crossover_comparisons.csv)
- [Held-out baseline comparisons](results/heldout.csv) · [Paired indicator comparisons](results/heldout_pairwise.csv) · [Fold scores](results/heldout_folds.csv)
- [New q B/D/G heatmaps](results/figures/q_heatmaps/benefit_degradation.png)
- Reused [width heatmaps](../distance_width/results/figures/A_BDG_heatmaps.png) and [r heatmaps](../r_rot/results/figures/A_distance_r_heatmaps.png). Known-bin cells equal the common-subset values; prior missing groups remain clearly separate. Those files are not modified.
- Parallel statistical D/G curves: [width](results/figures/width_conditional_DG.png), [q](results/figures/q_conditional_DG.png), [r](results/figures/r_conditional_DG.png). Shading is scene CI95; whiskers are seed SD; dotted lines flag limited support; panel scales differ.
- [Endpoint contrast CIs](results/figures/conditional_endpoint_contrasts.png) · [Held-out comparison](results/figures/heldout_comparison.png)

PNG/PDF/SVG exports accompany the new figures. All six PNGs regenerate identically in the recorded plotting environment; PDF/SVG timestamp/ID bytes are not asserted identical. Runtime used existing local CPU only, with added paid-resource expenditure $0. No Pods were created, restarted or modified. No scientific outputs outside this new directory are changed.
