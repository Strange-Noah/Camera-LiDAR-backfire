#!/bin/bash
# 第 5 段:补 MarkupSafe → 装 torch → 升级 setuptools → mim 三件套 → mmdet3d editable → 编译 BEVFusion
exec > /root/autodl-tmp/setup5.log 2>&1
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bevfusion

pip download MarkupSafe --no-deps -d /root/wheels
pip install --no-index --find-links /root/wheels torch==2.1.1 torchvision==0.16.1
python -c "import torch; print('TORCH_OK', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1

pip install "setuptools==69.5.1"

mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"

cd /root/mmdetection3d
pip install -v -e . --no-build-isolation
python -c "import mmdet3d; print('MMDET3D_OK', mmdet3d.__version__)" || exit 1

pip install "numpy==1.23.5"

export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
python projects/BEVFusion/setup.py develop

python -c "
import torch, mmdet3d
print('IMPORT_OK torch', torch.__version__, 'cuda', torch.version.cuda, '| mmdet3d', mmdet3d.__version__)
import mmcv, mmengine, mmdet, spconv
print('MMCV', mmcv.__version__, 'MMENGINE', mmengine.__version__, 'MMDET', mmdet.__version__, 'SPCONV', spconv.__version__)
x = torch.randn(4, 4).cuda(); print('CUDA_MATMUL_OK', (x @ x).sum().item() != 0)
"
echo "SETUP5_ALL_DONE"
