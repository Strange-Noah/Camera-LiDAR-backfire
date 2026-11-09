# OCaMo full-method reconstruction

Executed status: [Gate A FAIL; Gate B not run](../../docs/ocamo/FULL_METHOD_RESULTS.md).
The source has now run on the fixed nuScenes population. Original protocol and
historical implementation status follow; execution is not convergence acceptance.

Read [method checklist and limitations](../../docs/ocamo/FULL_METHOD.md) first.
This clean path leaves `ocamo/tools/nuscenes_minimal.py` frozen. It is implemented
and synthetically tested, not yet real-data accepted. No GPU or new dependencies.

After explicit paid-compute authorization, use existing remote Python and data:

```sh
P=/workspace/staging/ocamo/venv-cpu/bin/python
U=/workspace/staging/ocamo/official
# REPO is a checkout containing this implementation and the frozen minimal file.
# Use new output names; never overwrite first-baseline or directional results.
$P "$REPO/ocamo/full_method/run.py" --baseline "$REPO" --upstream "$U" \
  --dataroot /workspace/datasets/nuscenes-validation --output fresh-inventory --inventory-only
# Inspect counts/missing files and fixed scene assignment before the full run.
$P "$REPO/ocamo/full_method/verify_runtime.py" \
  --dataroot /workspace/datasets/nuscenes-validation --output fresh-feature-check
# Review the fixed six private overlays; stop if geometry/features are invalid.
$P "$REPO/ocamo/full_method/run.py" --baseline "$REPO" --upstream "$U" \
  --dataroot /workspace/datasets/nuscenes-validation --output fresh-gate-a
$P "$REPO/ocamo/full_method/plot.py" --results fresh-gate-a --output fresh-gate-a-plots
```

Run focused tests from this directory with the intended CPU environment:

```sh
python -m unittest -v test_method
```

`inventory.json` contains real pairing/geometry and availability, `training.json`
automatic labels/features, `preselection.json` scene-disjoint model/CV/threshold,
`trajectories.jsonl` every method's observations and corrections, `summary.json`
all finals and prefixes. Files remain remote except small sanitized reports,
tables and plots. Full metadata/inventory and sensor overlays are not published.

The driver never starts resources, downloads data, installs packages, performs
BEVFusion inference or runs Gate B. `summary.json` always requires scientific
review; finite execution cannot automatically accept Gate A.
