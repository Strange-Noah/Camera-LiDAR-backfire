# Three geometric explanations of Spatial Backfire

**No indicator provides a stable, general explanation beyond distance in this fixed comparison.** Jacobian r_rot has the clearest local degradation contrast at ≥40m, which survives scene-plus-seed sensitivity. The physical q_rot is interpretable but does not outperform pixel width or remove distance dependence. None of the three improves the prespecified scene-held-out distance model with a positive two-sided CI. These are conditional associations and a deliberately simple prediction comparison, not causal or deployment results.

## Population and physical definition

All 53,043 accepted Car identities, the original 6,019-sample cohort and frozen 27 hit conditions are retained. Direct three-way statistics use the **same 52,664 valid-projection objects**, fixed at all σ/seeds. The other 379 objects have separate accepted B/D/G summaries and q-bin counts, not an implicit population intersection. Pixel widths, camera selection, distance and Jacobian r_rot are unchanged. Every populated width/r cell reproduces its accepted prior B/D/G and seed statistics.

The previously used official nuScenes trainval metadata URL was available without a paid machine. Six required annotation/pose tables were read locally; all accepted tokens, translations, instance/scene identities and distances reconcile. No projection, inference, E4 matching or dataset audit was rerun. Input hashes are retained in [source provenance](results/metadata_source_hashes.json). No per-object metadata is published.

Let e be the horizontal LIDAR_TOP ego-position-to-GT ray at the sample's original LiDAR keyframe, and u its perpendicular horizontal unit vector. With GT box rotation R and nuScenes dimensions [width,length,height], define

`W_phys = length·|uᵀR[:,0]| + width·|uᵀR[:,1]| + height·|uᵀR[:,2]|`.

This is the full box span transverse to that ground-plane viewing ray; it includes orientation and any height contribution under roll/pitch. It is not pixel width or an inferred physical dimension. The single predefined proxy is `q_rot5 = radians(5)·Z/W_phys`, with the accepted ground-plane ego distance Z (not camera optical depth). Its bins <0.1, 0.1–0.5, 0.5–1 and ≥1 were fixed before computing new q outcomes, matching the existing r bin edges. It is a one-angle lateral approximation, not the three-axis Gaussian projection RMS measured by r_rot. No coefficient, width convention or bins were optimized.

## Unified statistical comparison

B=LiDAR miss−clean miss; D=noisy miss−clean miss; G=D−B, all in percentage points. Complete σ1…5 curves, all five seed values, mean/sample SD, observed upward brackets and B/D/G scene intervals are retained. The primary curve summary below is the arithmetic mean D over the five observed σ levels; it is not a continuous-dose integral.

Following [Temporal's paired scene/trend approach](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/nuo-ft-conditional-threshold/scripts/temporal/conditional_threshold), 500 complete-scene multinomial draws are shared across every variable/cell. Support uses the established Spatial rule N≥200 and ≥10 scenes; this differs from Temporal's trajectory/top-scene rule. Each supported adjacent contrast and risk-ordered endpoint contrast is reported. Expected risk increases with decreasing pixel width or increasing q/r; residual-distance contrasts increase with distance. Strata with fewer than two supported cells are not testable.

A supplemental two-level sensitivity uses the same scene draw plus five seed indices sampled with replacement independently at each σ, shared across all compared cells. This preserves pairing, never treats seeds as GT observations, and only resamples the five realized rotations. It is not full perturbation-distribution uncertainty. Bootstrap direction/order shares are descriptive frequencies, not posterior probabilities or p-values. A ≥0.95 share is the Temporal-style directional convention, distinct from a two-sided CI excluding zero. Intervals and trends are pointwise/exploratory, with no multiplicity adjustment; Monte Carlo resolution is 1/500.

## 1. Does projected width retain statistically supported information beyond distance?

**Some localized, nonmonotonic separation is supported; a general inverse-width explanation is not.** At ≥40m, <50px minus 50–100px mean-curve D is **−4.539pp**, scene CI95 **[−6.712,−2.324]**, and scene-plus-seed CI **[−7.668,−1.304]**. The narrowest group degrades less, opposite the hypothesized risk direction. This reversal is not explained by scene/observed-seed resampling alone. Other irregularities, such as the 20–30m supported endpoint contrast −0.587pp [−1.429,0.373], remain consistent with either direction under sampling variability.

Every supported within-distance *endpoint* mean-D CI includes zero for width. Strict inverse-width mean-curve ordering has scene support 0.504 / 0.008 / 0.074 / 0.000 across 10–20 / 20–30 / 30–40 / ≥40m. The near stratum has only one supported width cell. This does not erase individual contrasts, nor imply that all width information is absent.

## 2. Is q_rot better than projected width?

**Not established.** q's supported endpoint mean-D estimates are positive at 10–20m (0.326pp [−0.026,0.756]), 20–30m (0.790pp [−0.147,1.822]) and 30–40m (1.661pp [−2.643,7.203]), but all two-sided CIs include zero. The first two positive-direction shares are 0.960/0.964, falling to 0.940/0.946 under scene-plus-seed resampling. This is suggestive directional evidence, not consistent two-sided support.

At ≥40m, higher q has **less** mean-curve degradation (−0.638pp [−2.698,1.348]); D5 is nearly equal (18.667 vs 18.517pp). Strict positive q ordering is therefore not universal. The 30–40m lowest supported q cell has 224 objects across 37 scenes, with very wide G intervals; passing a minimal support cutoff does not ensure precision. No sparse-cell recovery or bin changes were made.

In paired held-out scoring, adding q instead of width changes MSE by an improvement of **−0.179pp² [−0.671,0.343]**; no advantage is demonstrated. q is a physically motivated proxy, but its chosen physical span omits camera geometry and occlusion, and remains strongly related to range.

## 3. Does the Jacobian r_rot add information beyond q_rot?

**There is a stable far-distance descriptive contrast, but no demonstrated incremental predictive gain beyond q in the fixed additive comparison.** At ≥40m, r≥1 minus 0.5≤r<1 gives mean-curve D **+2.770pp [0.593,5.105]**, positive share **0.990**; the scene-plus-seed interval is **[0.611,5.160]**, share **0.990**. G5 differs by **+4.413pp [1.079,7.237]**, surviving the seed sensitivity [0.900,7.663].

At 20–30m and 30–40m, r endpoint mean-D differences are 0.007pp [−1.481,1.308] and −0.315pp [−1.606,1.069]. These weak/irregular means are consistent with scene and seed variation. At 10–20m the endpoint difference is 0.683pp [−0.241,2.110]; its highest-r cell has only 259 objects. Near low-r groups have little or negative initial benefit.

Adding r to distance+q gives an MSE improvement of **−0.031pp² [−0.066,−0.002]**: a tiny worsening in this particular fixed held-out comparison. Comparing distance+r with distance+q gives **+0.018pp² [−0.059,0.101]**, also inconclusive. This does not establish that r has no information or is intrinsically inferior; fixed main effects cannot express arbitrary interactions, and local r/q partitions differ.

## 4. Which has the most stable degradation ordering?

**No global winner; r has the clearest supported local positive contrast at ≥40m.** At that distance there are only two supported r groups, and their positive mean-D ordering holds in 99% of both bootstrap variants. q's strongest positive ordering is at 10–20m with two supported groups (0.960 scene; 0.940 scene-plus-seed). Width has a stable reversed adjacent contrast rather than stable expected ordering. Counts, available groups and endpoint ranges differ; these frequencies cannot be treated as a calibrated feature-importance ranking.

The [compact cross-variable table](results/cross_variable_comparison.csv), [all trends](results/trends.csv) and [all contrasts](results/contrasts.csv) show every σ, both uncertainty modes and residual-distance direction. Conclusions do not require strict monotonicity in the observed means.

## 5. Which gives useful Backfire/crossover separation?

**r and q each show local G separation at ≥40m; neither yields a generally reliable earlier-crossover ordering.** For q≥1 versus 0.5–1, G5 is **+3.754pp [1.726,5.603]**, with sensitivity CI **[0.342,7.459]**. However, B falls from **10.289 to 6.686pp**, while D5 barely changes. This G separation mainly reflects less initial fusion benefit rather than greater rotation-induced degradation. r's far-group G separation accompanies larger D as well.

For crossover comparisons, retain the accepted observed brackets and additionally record the first observed σ that remains positive through σ5. Clean benefit must be positive; no crossing through 5° is right-censored, not a measured 6° threshold. Bootstrap earlier/tie/later shares are conditional on both clean benefits remaining positive, with the eligibility fraction reported. There is no interpolation or CI on an artificial censor code.

At ≥40m, the high-r endpoint is earlier in **65.8%** of eligible scene draws and tied in 34.2%; high-q is earlier in **13.2%**, tied in 86.8%. Width's narrow endpoint is earlier in 42.5%. These do not support a ≥95% earlier-onset claim. At 10–20m/width, only 38.2% of draws have positive clean benefit at both endpoints; treating every draw as a newly induced transition would be misleading. Complete conditional crossover distributions and order shares are retained in the tables. G separation at a dose, D prediction and crossover prediction are different questions; the held-out exercise evaluates D only.

## 6. How much residual distance association remains?

**Substantial association remains for all three indicators.** Supported far-minus-near mean-curve D contrasts include:

| Fixed indicator bin | Distance endpoints | Difference, pp | Paired scene CI95 |
|---|---|---:|---|
| Pixel width 100–200px | ≥40 minus 10–20m | 13.567 | [12.171,14.923] |
| Pixel width ≥200px | ≥40 minus 0–10m | 8.642 | [3.866,14.165] |
| q 0.5–1 | ≥40 minus 10–20m | 12.666 | [11.149,14.198] |
| q ≥1 | ≥40 minus 20–30m | 10.195 | [8.824,11.600] |
| r 0.5–1 | ≥40 minus 10–20m | 9.982 | [7.583,12.241] |
| r ≥1 | ≥40 minus 10–20m | 12.288 | [10.471,13.645] |

Each displayed residual contrast has positive scene support 1.000. Endpoint distances differ because support differs; these are not equal-exposure standardized effects. Coarse bins leave continuous within-bin geometry, point support, camera selection and scene composition uncontrolled.

## Incremental explanatory comparison

Five scene-held-out folds use the same 52,664 objects, fixed splits and all five D targets (each target is the five-seed mean at one σ). Distance-only OLS uses five bin indicators. Additive models add three independent columns for each four-bin geometric variable. No interaction, high-capacity model, regularization or parameter search is used. MSE is per-object squared error averaged over the five σ levels in pp²; uncertainty resamples paired scene totals of fixed out-of-fold errors, without refitting the regressions inside the bootstrap.

| Added term | Reference | MSE improvement, pp² | Scene CI95 |
|---|---|---:|---|
| width | distance | +0.104 | [−0.373,0.540] |
| q | distance | −0.075 | [−0.177,0.020] |
| r | distance | −0.057 | [−0.143,0.015] |
| r | distance+q | −0.031 | [−0.066,−0.002] |

Distance-only MSE is 263.316pp². Improvements are tiny relative to this noisy object-level target. None has a positive two-sided CI. This fixed comparison does not support a general out-of-scene predictive improvement, nor rule out a different prespecified scientific model in future work. It does not estimate causal effects or prediction performance for unseen rotation distributions. [All pairwise scores](results/heldout_pairwise.csv) and fold-level scores are retained.

## 7. Is this comparable to Temporal r500?

The **methodological framework transfers**: fixed groups, B/D/G decomposition, distance controls, paired scene contrasts, trend frequencies and explicit support. The strength of Temporal's reported normalized-displacement explanation does **not** transfer here as an established universal Spatial mechanism. q is dimensionless and physically motivated, but is a ground-ray one-angle approximation; r uses three-axis first-order camera RMS sensitivity; pixel width is scale alone. None removes range dependence or achieves consistent conditional crossover ordering. Temporal also used different populations, models, interventions and interpolated delay thresholds, so its numerical thresholds cannot be equated with these rotational grid brackets.

[Reproduction and output map](README.md) · [Focused validation](VALIDATION.md). No previous result is rewritten. This comparison ends here; no further feature definitions, bin search, unrelated analyses or Robust Fusion development are undertaken.
