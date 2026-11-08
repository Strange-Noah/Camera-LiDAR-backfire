# Verified remote-first CPU execution

The tested runtime is Ubuntu24.04 / Python3.12.3 on2vCPU/4GB, with dependencies pinned in `ocamo/requirements-cpu.lock.txt`. Neither CARLA, nuScenes nor checkpoints are copied to the Mac. Do not change the BEVFusion environment.

On an explicitly authorized remote CPU environment, the setup is:

```sh
python3 -m venv /workspace/staging/ocamo/venv-cpu
/workspace/staging/ocamo/venv-cpu/bin/python -m pip install -r ocamo/requirements-cpu.lock.txt
```

Official source: https://github.com/moravecj/OCaMo pinned to `1ac55de3c0eabb02f155cd44f7481117536c9526`, respecting its CC BY-NC4.0 license and paper attribution. Place it under `/workspace/staging/ocamo/official` and retain its original layout. Its CARLA archive is now present remotely at `/workspace/staging/ocamo/carla_example.zip` (722112251 bytes), extracted under `official/example/carla_example` (400 files,961343167 uncompressed bytes). Check before downloading again. This session's download was explicitly authorized; future large downloads still require approval.

Execute from this repository using fresh output paths:

```sh
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/ocamo-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
P=/workspace/staging/ocamo/venv-cpu/bin/python
U=/workspace/staging/ocamo/official
R=/workspace/results/ocamo/YOUR-FRESH-RUN-ID
$P ocamo/tools/run_official.py --upstream "$U" --output "$R/ocamo-drift-1" --script ocamo_drift_tracking
$P ocamo/tools/run_official.py --upstream "$U" --output "$R/lto-drift-1" --script lto_drift_tracking
$P ocamo/tools/run_official.py --upstream "$U" --output "$R/ocamo-monitoring-1" --script ocamo_monitoring
$P ocamo/tools/run_official.py --upstream "$U" --output "$R/lto-monitoring-1" --script lto_monitoring
$P ocamo/tools/nuscenes_minimal.py --upstream "$U" --dataroot /workspace/datasets/nuscenes-validation --output "$R/nuscenes-3scenes-2seeds" --scenes 3 --seeds 0 1 --sigma .5
# Exact-repeat evidence uses fresh sibling directories:
$P ocamo/tools/run_official.py --upstream "$U" --output "$R/ocamo-drift-2" --script ocamo_drift_tracking
$P ocamo/tools/run_official.py --upstream "$U" --output "$R/lto-drift-2" --script lto_drift_tracking
$P ocamo/tools/nuscenes_minimal.py --upstream "$U" --dataroot /workspace/datasets/nuscenes-validation --output "$R/nuscenes-repeat" --scenes 3 --seeds 0 1 --sigma .5
$P ocamo/tools/check_run.py --results "$R"
```

`run_official.py` runs original files with runpy from their required working directory. It changes only figure output and progress logging of unchanged MAT reads. No loss, feature extractor, random seed, parameters or optimizer rules are altered. Official arrays and plots are saved; SGDSchaul burn-in is10, LTO has no burn-in. Python demonstration performance is not full MATLAB/paper benchmark reproduction.

The validation root contains genuine samples, sweeps and metadata. The initial adapter uses only camera/LiDAR keyframes, sorted by real camera timestamps, not high-frequency sweeps. First3 lexicographically sorted scene tokens are fixed independent of outcomes; each has40 usable keyframes. Scenes and seed selection must not be retuned to hide failures. Original larger data and checkpoints remain in their existing remote directories.

The completed checkpoint is `/workspace/results/ocamo/2026-11-09-first-baseline`; do not overwrite it. Its saved run directories are `ocamo-drift-1`, `ocamo-drift-2`, `lto-drift-1`, `lto-drift-2`, `ocamo-monitoring-1`, `lto-monitoring-1`, `nuscenes-3scenes-2seeds`, and `nuscenes-repeat`. Raw trajectories, projections and correction matrices stay remote. Small aggregate reports and line graphs may be published. Run `check_run.py --results "$R"` there to validate saved paired results; it does not run experiments.

Resource limits belong to each explicit authorization. Do not start or create resources from these commands, and do not assume a previous spend authorization still applies. No GPU inference or weighting is part of this baseline.
