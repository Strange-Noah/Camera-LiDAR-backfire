# Focused Distance × projected-width extension

[Scientific findings](DISTANCE_WIDTH_FINDINGS.md) · [Validation](VALIDATION.md) · [Tables and four figures](results)

This additive extension uses the accepted fixed distance and pixel-width bins from `../screening/bins.json`. It reuses `screening.Data`, its identity checks, mask hashes, CSV schema and `core.summarize` (500 paired complete-scene bootstrap replicates, fixed seed). Scene-only G intervals condition on the five observed rotation seeds. Sample SD across seeds is retained separately; seeds are not independent GT observations. Limited support means N<200 or fewer than 10 scenes, matching the Spatial screening convention rather than asserting Temporal's trajectory-support rule.

Use the existing plotting environment specified by `../../evaluation/plot-requirements.txt`. No geometry-runtime or GPU environment is needed because clean geometry is frozen. Inputs are the accepted extraction directory (`objects.json`, condition `summary.json`, 27 hit NPZs), accepted `geometry.json.gz`, and existing `results/metadata_audit/independence/scene_support.csv` from the repository root. The shared loader reads the scene table; no motion analysis or extra feature extraction is performed. Private per-object caches are not distributed.

From this directory:

```sh
python analyze.py --labels ACCEPTED_EXTRACTION --geometry ACCEPTED_GEOMETRY.json.gz --scenes SCENE_SUPPORT.csv --output fresh-results
python analyze.py --summary results/summary.json --output fresh-figures
python validate.py --summary results/summary.json --phase-a ../screening/phase_a/summary.json --labels ACCEPTED_EXTRACTION --geometry ACCEPTED_GEOMETRY.json.gz --scenes SCENE_SUPPORT.csv --regenerated fresh-figures
```

Output directories must be fresh. The first command groups labels without inference or matching. The second regenerates all four figures solely from the aggregate JSON; PNGs are reproducible in the same plotting environment (PDF/SVG timestamps/IDs may differ). The third verifies raw fixed-mask rates when input caches are supplied, numerical identities, seed aggregation, crossings, support, both Phase A marginals and all four PNGs. Without raw inputs, aggregate checks remain usable but raw membership is reported NOT_RUN.

Distance bins are [0,10), [10,20), [20,30), [30,40), ≥40m; width bins are <50, [50,100), [100,200), ≥200px. Missing widths remain a separate group. Nothing is imputed or clipped. B/D/G and all 27 original miss rates are in percent/percentage points. Arrays D/G are σ1…5 by seeds0…4. Clean B is deterministic in these frozen labels. Figure B shows clean G=−B at σ=0 and observed mean±sample seed SD at σ1…5. Curves connect measured grid points; no interpolation-based onset is claimed.

Execution is local CPU only, added paid-resource cost $0. The task ends with the five scientific answers; no additional pairs or Robust Fusion development are included.
