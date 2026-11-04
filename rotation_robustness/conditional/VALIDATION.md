# Scientific acceptance

**PASS for the specified Car distance and LiDAR-point analysis.** No GPU inference or official nuScenes scoring was run.

| Gate | Outcome |
|---|---|
| Conditions | 25 rotation conditions, five seeds at all five levels, plus accepted clean/LiDAR references |
| Population | 6,019 samples; 53,043 eligible Car annotations; no intersection reduction |
| Identity | All 27 formatted JSON SHA256 values checked; source archive receipts and retained shards present; frozen infos and 13 metadata tables verified |
| Matching | Corrected all-unmatched-GT rule; actual Temporal conditional float64 matcher; 100 deterministic parity fixtures including ambiguous assignments |
| Object alignment | Unique sample+annotation keys; exact class and global coordinates; unambiguous annotation association; GT-only subgroup membership |
| Predictions | Sample membership, shared population, finite positions/scores and duplicate predictions checked; original accepted formatter evidence covers dimensions/class fields |
| Denominator | Identical 53,043 GT observations across all conditions; all 20 subgroup count totals independently reconcile to retained per-object hits |
| Numerical | G=D−B; individual seeds retained; all 27 hit-cache hashes verified |
| Uncertainty | Seed sample SD plus 500 paired complete-scene bootstrap replicates, fixed seed; scene intervals conditional on observed seeds |
| Figures | Three real-data figures, PNG/PDF/SVG; visually inspected; regenerated from published scene counts |
| Cross-environment reproducibility | All point results agree; maximum CI rounding difference 1.78e−15 percentage points (tolerance 1e−10) |
| Preservation | Previously published rotation artifacts unchanged; no new predictions or scientific manifests |
| Cleanup | Task CPU deleted; account-level GET 404 confirmed |

The matching extraction ran in the original frozen CPU runtime; summary/plot regeneration used the retained plotting environment in the parent evaluation requirements. The initial exact-equality cross-environment check failed only on machine-precision bootstrap percentile rounding; tolerance-based reconciliation passed and that discrepancy is retained here.

These are descriptive five-seed mean crossings, not universal or statistically established physical thresholds. Near-cell camera benefit can be negligible. Cells with B≤0 are not newly induced transitions. Sparse cells are marked and receive no robust-effect interpretation. LiDAR-point associations do not isolate a causal mechanism. Scene-only intervals do not represent seed-population uncertainty.

Temporal source: [conditional analysis](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/nuo-ft-conditional-threshold/scripts/temporal/conditional_threshold). Its GT is derived from detection-evaluation filtering. Its population/model differ, and its older corrected E4 script used float32 while the actual conditional matcher uses float64. This study reproduces the latter; it does not claim bitwise equivalence to the older float32 audit. Temporal's unresolved FP-file discrepancy is irrelevant to interpreting this study as an FP analysis: no such FP claim is made.

The extraction ledger and small per-condition hit caches remain durable internal evidence. Public scene counts suffice to reproduce summaries and plots; no private paths, credentials, infrastructure journals or raw predictions are distributed. Extraction resume validates identities, denominator and cache hashes; full fault-injection testing of resume was not performed during this run.
