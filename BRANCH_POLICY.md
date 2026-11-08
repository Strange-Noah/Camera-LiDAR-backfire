# OCaMo branch policy

- noah/ocamo-baseline: dependencies, faithful official reproduction, nuScenes adapter, persistent unweighted online estimates, convergence and correction export.
- noah/ocamo-experiments: reserved for later explicitly authorized weighting and robustness experiments. No weighting is implemented or authorized now.

The baseline foundation is f9acebc5b39076138b656315476a667f339d6109. Existing environment-baseline, robustness-experiments, main, teammate branches and the closed local preflight branch are read-only. No new task branches, rebases, resets, force-pushes or automatic experimental back-merges. Integrate verified baseline updates deliberately with recorded SHAs and regression evidence.
