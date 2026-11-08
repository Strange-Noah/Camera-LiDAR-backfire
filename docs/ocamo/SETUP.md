# Minimal remote-first execution path

**Not executed.** Dependencies below are a proposal, not a tested version lock. Keep the established BEVFusion environment unchanged. A separate remote Python3.12 CPU venv is the preferred starting candidate once an execution channel is authorized and available. The official examples do not require PyTorch.

One setup command, only after authorization, from the repository root:

```sh
python3.12 -m venv .venv-ocamo && .venv-ocamo/bin/python -m pip install -r ocamo/requirements-cpu.in
```

Then record `python -V`, `pip freeze`, platform and import checks; retain a tested exact dependency lock after successful execution. Do not install now simply to make an audit pass.

Pin official source to https://github.com/moravecj/OCaMo commit `1ac55de3c0eabb02f155cd44f7481117536c9526`. Preserve the original directory layout and example paths. Set OCAMO_ROOT to that checkout and OCAMO_PYTHON to the absolute isolated interpreter in the remote workspace. With author data extracted under `$OCAMO_ROOT/example/carla_example`, the intended first commands are:

```sh
cd "$OCAMO_ROOT/python/inference"
MPLBACKEND=Agg "$OCAMO_PYTHON" ocamo_drift_tracking.py
MPLBACKEND=Agg "$OCAMO_PYTHON" lto_drift_tracking.py
```

These commands are source-derived, not verified runs. Upstream calls plt.show without exporting numerical trajectories: a clearly isolated execution wrapper will be needed to persist original arrays and plots, without changing objective/update behavior. Monitoring follows only after both drift examples succeed. No numerical tolerance has yet been validated against author outputs.

Author-linked CARLA download: https://cmp.felk.cvut.cz/~moravj34/data/carla_example.zip — HEAD200, Content-Length722112251 bytes (about689MiB). First check persistent storage for an existing archive/extracted copy. If absent, request permission to download exactly once remotely to `/workspace/staging/ocamo/carla_example.zip`; extract to a versioned remote source checkout's `example/carla_example`. Neither archive nor sensors may be downloaded to the Mac. The archive's uncompressed size is currently unknown, so do not claim a total storage requirement. No archive download is authorized or performed.

Previously documented data locations to verify, not asserted currently readable: `/workspace/datasets/nuscenes`, `/workspace/checkpoints`, `/workspace/results`, `/workspace/staging`. Use actual sample_data filenames/timestamps once access is available. Do not assume keyframe-only coverage implies sweeps exist, or confuse metadata presence with sensor availability. Do not copy checkpoints into this repository.
