#!/bin/bash
# BEVFusion baseline 环境搭建(AutoDL, RTX 4090, CUDA 12.1 driver / cu118 wheels)
exec > /root/autodl-tmp/setup.log 2>&1
set -x

# ---- 1. 解压 mini 数据集(约 4GB)----
mkdir -p /root/autodl-tmp/nuscenes_mini
tar -xzf /root/autodl-pub/nuScenes/Fulldatasetv1.0/Mini/v1.0-mini.tgz -C /root/autodl-tmp/nuscenes_mini
ls /root/autodl-tmp/nuscenes_mini/
echo "MINI_EXTRACT_DONE"

# ---- 2. conda 环境 ----
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n bevfusion python=3.10 -y
conda activate bevfusion

# ---- 3. PyTorch cu118 ----
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print('TORCH_OK', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- 4. OpenMMLab ----
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
mim install "mmdet==3.2.0"

# ---- 5. mmdet3d v1.4.0 ----
cd /root
git clone --depth 1 --branch v1.4.0 https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e .

# ---- 6. 稀疏卷积 + 依赖锁定 ----
pip install cumm-cu118 spconv-cu118
pip install nuscenes-devkit numpy==1.23.5 yapf==0.40.1 setuptools==58.0.4

# ---- 7. 编译 BEVFusion 算子 ----
python projects/BEVFusion/setup.py develop

# ---- 8. 验证 ----
python -c "import mmdet3d, spconv, torch; print('IMPORT_OK mmdet3d', mmdet3d.__version__)"
echo "SETUP_ALL_DONE"
