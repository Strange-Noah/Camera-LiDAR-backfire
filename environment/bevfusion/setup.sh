#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python -c 'import sys; assert sys.version_info[:2] == (3,8), "Use the isolated Python 3.8 environment"'
nvcc --version | grep 'release 11.3'
python -m pip install pip==23.3.2 setuptools==59.5.0 wheel==0.38.4
python -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html
python -m pip install -r "$here/requirements/runtime.txt"
# mmcv-full provides the mmcv namespace: do not install both distributions.
python -m pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
commit=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$here/upstream.json")
if [[ ! -d /opt/bevfusion/.git ]]; then
  git clone https://github.com/mit-han-lab/bevfusion.git /opt/bevfusion
fi
git -C /opt/bevfusion checkout --detach "$commit"
test -z "$(git -C /opt/bevfusion status --porcelain)"
cd /opt/bevfusion
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.0;8.6+PTX' MAX_JOBS=${MAX_JOBS:-4}
python setup.py develop
python -m pip check
python -m pip freeze > /opt/deployment/bevfusion/installed.freeze.txt
