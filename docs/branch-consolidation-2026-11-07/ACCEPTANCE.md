# Branch consolidation acceptance — 2026-11-07

**PASS — history-preserving integration; no scientific content changes, reruns or paid resources.**

## Sources and recoverable backups

| Source | Original immutable commit SHA | Annotated backup tag |
|---|---|---|
| noah/rotation-robustness-results | [f3fa506580e227c82fd264df8628dfe4f76cbce6](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/f3fa506580e227c82fd264df8628dfe4f76cbce6) | [archive/consolidation-2026-11-07/noah/rotation-robustness-results](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/archive/consolidation-2026-11-07/noah/rotation-robustness-results) |
| noah/spatial-rrot-analysis | [62cbe9dc83f61a5dd71b8661b7ea13d00227eac9](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/62cbe9dc83f61a5dd71b8661b7ea13d00227eac9) | [archive/consolidation-2026-11-07/noah/spatial-rrot-analysis](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/archive/consolidation-2026-11-07/noah/spatial-rrot-analysis) |
| noah/spatial-conditional-backfire | [afa2b8228ce1f28d876ef6783e5380d05f6aee8d](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/afa2b8228ce1f28d876ef6783e5380d05f6aee8d) | [archive/consolidation-2026-11-07/noah/spatial-conditional-backfire](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/archive/consolidation-2026-11-07/noah/spatial-conditional-backfire) |
| noah/spatial-conditional-screening | [2b68b92c620f11626242caa178626c8ed39bb718](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/2b68b92c620f11626242caa178626c8ed39bb718) | [archive/consolidation-2026-11-07/noah/spatial-conditional-screening](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/archive/consolidation-2026-11-07/noah/spatial-conditional-screening) |
| feat/bevfusion-mini-deployment | [24764baafda8aa48ede2cc9f3138b883f8482520](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/24764baafda8aa48ede2cc9f3138b883f8482520) | [archive/consolidation-2026-11-07/feat/bevfusion-mini-deployment](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/archive/consolidation-2026-11-07/feat/bevfusion-mini-deployment) |
| feat/validation-baseline-preparation | [823d8a69a0c6c3032a13f6af269656f27f4dbfca](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/823d8a69a0c6c3032a13f6af269656f27f4dbfca) | [archive/consolidation-2026-11-07/feat/validation-baseline-preparation](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/archive/consolidation-2026-11-07/feat/validation-baseline-preparation) |
| chore/independent-compute-workflow | [9c4c32a3224d6338ce7df21416214fe979652465](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/9c4c32a3224d6338ce7df21416214fe979652465) | [archive/consolidation-2026-11-07/chore/independent-compute-workflow](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/archive/consolidation-2026-11-07/chore/independent-compute-workflow) |

All seven branch-exclusive histories were authored by Strange-Noah (same verified Git identity as current personal work). Environment mini is ancestor of validation; conditional-backfire is ancestor of rrot, which is ancestor of screening. Rotation-results and independent-compute each retain their independent commits through ordinary two-parent merges. Full exact merge bases, unique commit lists and ownership evidence are in [evidence.json](evidence.json).

## Targets and content verification

- [noah/robustness-experiments](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/noah/robustness-experiments): integration [3c7239bbf9866a1d5b5fc352c7ad9b83126c6158](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/3c7239bbf9866a1d5b5fc352c7ad9b83126c6158); tree `d079c3ff37c4e3fc2e4738f36c82d2d5ac0b054a`. All assigned source tips are ancestors.
- [noah/environment-baseline](https://github.com/Strange-Noah/Camera-LiDAR-backfire/tree/noah/environment-baseline): integration [78901cbe634e5e7f3d5138a039dab1de75731030](https://github.com/Strange-Noah/Camera-LiDAR-backfire/commit/78901cbe634e5e7f3d5138a039dab1de75731030); tree `87ee7afe7ba56cfcea5ee1723ffb0fd41da36d6b`. All assigned source tips are ancestors.
- Final baseline organization commit: `f9acebc5b39076138b656315476a667f339d6109`. The experiment branch adds this report, machine-readable evidence and BRANCH_POLICY.md after its integration commit. Its final commit is available via this document’s GitHub commit/permalink and the final handoff.

The experiment integration tree is byte-for-byte identical to screening: rotation pipeline/configurations, accepted spatial/conditional/rrot/width/geometric reports and all retained research files are preserved. Baseline integration is byte-for-byte identical to validation except three additive workflow files: CONTRIBUTING.md, configs/experiment-record.example.json, docs/repository_layout.md. Mini histories and earlier baseline scripts remain reachable. Environment instructions, deployment documentation, original baseline validation artifacts and tests remain exact Git blobs. No tests or experiments were rerun for a reference-only organization change.

No Conditional Weighting implementation was found in the baseline target; no experiment source was merged into it. Original OCaMo/LTO calibration implementation is not present in these historical source branches; this task preserves existing environment/baseline artifacts and establishes the future boundary, rather than claiming or creating missing algorithms.

Merge conflicts: none. No manual conflict resolutions, rebases, squashes, force-pushes or rewrites. An initial overly strict every-historical-blob equality check rejected three intentionally superseded historical files; the correct latest-source-plus-independent-delta check passed, with all historical versions retained by ancestry and backup.

## Protected references and cleanup

Remote main, jhe-sudo/temporal-misalignment, nuo-ft-conditional-threshold and refactor/windows-local-setup exact pre/post SHAs are recorded in evidence.json. Only seven explicitly audited personal source heads are cleanup candidates. Backups were pushed and independently checked before deletion; targets were published and checked before cleanup. No teammate head is written or deleted.

Restore any source with `git fetch origin --tags` followed by `git branch ORIGINAL_NAME archive/consolidation-2026-11-07/ORIGINAL_NAME`; push that branch normally if remote restoration is needed. Backup tags must never be moved or deleted. The SHA column also allows exact recovery independent of a branch name.

See [BRANCH_POLICY.md](../../BRANCH_POLICY.md) for future routing. No per-task branches or automatic experiment-to-baseline merges.

Final cleanup: **PASS**. All seven audited source heads were removed only after independent remote verification. The remote now contains exactly the two personal target heads, main and the three unchanged teammate heads. Exact protected post-cleanup SHAs are in evidence.json. Current local work is bound to noah/robustness-experiments; pre-existing uncommitted work remains intact.
