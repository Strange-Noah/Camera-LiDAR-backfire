# Lightweight acceptance

**Phase A PASS for the eight locally available diagnostics.** Metadata coverage limitations are explicit in the findings and README.

- All 53,043 unique sample+annotation identities align with the existing geometry records; no E4 labels changed or matching repeated.
- Each feature partitions the entire accepted Car denominator, including explicit missing groups. Fixed masks apply to all 27 conditions.
- The existing summarizer retains per-seed rates, B/D/G, means, sample SD and paired scene-only bootstrap intervals.
- G=D−B is checked to 1e−10pp for every populated group. Seed means/SD and the accepted distance-only summary reconcile.
- The scene-support table has unique scene keys and covers all 150 target scenes. Its features are labeled as scene aggregates, not instantaneous motion.
- Bin boundaries precede new outcome inspection. No bin or normalization search occurred.
- Empty groups are retained; limited-support groups are flagged. Baseline-no-benefit and small-benefit flags prevent misleading early-crossing interpretations.
- Final tables regenerate the figures. Axes use rotational σ and object-miss percentage points, not mAP/NDS.
- No paid resources, new scientific predictions, official scoring, training or rare-class recovery.

These are exploratory descriptive comparisons. Different binnings/group counts prevent interpreting the range table as calibrated feature importance. Scene-only intervals do not cover seed-population variability. No multiple-testing or causal claim is made.

Phase B pair 1 (fixed r_rot × LiDAR support): PASS. All cells plus missing groups sum to 53,043; fixed masks and all 27 rates/seed values retained; sparse cells flagged.

## Final focused checks

**Phase B pair 2 (Distance × selected camera): PASS.** All 30 known-camera/distance cells satisfy N≥200 and ≥10 contributing scenes; missing and empty cells remain explicit. No denominator restriction was introduced.

For both new pairs:

- All cells total 53,043. Both axis marginal counts and weighted LiDAR/clean/noisy rates, B, D and G reconcile with the corresponding Phase A results to 1e−10pp.
- Every populated cell satisfies G=D−B to 1e−10pp. Independently recalculated means and sample SD across the five seed values agree to 1e−12pp.
- Figures regenerated from the published summary JSON are byte-identical PNGs, including the unchanged first-pair figure. The final camera figure was visually checked after abbreviating camera labels; the findings define the abbreviations. Full seed values and scene intervals remain available in tables/JSON.
- Source syntax, relative document links and public artifact private-path/credential-pattern scans pass. No private per-object cache or internal audit log is included in this publication.
- Phase A was published in `3bc7b8d`; pair 1 in `3e56c55`; pair 2 was calculated after pair 1 publication. Bins and accepted prior analysis directories were preserved. New work is additive under `screening/` on the personal branch.

Execution used the existing local CPU plotting environment. Added paid-resource spend is zero. Metadata gaps remain explicitly unavailable, not a failed numerical check. The acceptance covers the descriptive statistics and reproducible plots; causal identification, deployment-time proxies and Robust Fusion interventions remain outside scope.
