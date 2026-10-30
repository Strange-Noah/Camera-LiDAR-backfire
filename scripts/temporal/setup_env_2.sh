#!/bin/bash
# 第 2 段:torch 走 wget 直链,然后继续 OpenMMLab 全家桶 + 权重下载
exec > /root/autodl-tmp/setup2.log 2>&1
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bevfusion

# ---- torch / torchvision:先 wget 直链再本地安装 ----
mkdir -p /root/wheels && cd /root/wheels
wget -c -q https://download.pytorch.org/whl/cu118/torch-2.1.1%2Bcu118-cp310-cp310-linux_x86_64.whl
wget -c -q https://download.pytorch.org/whl/cu118/torchvision-0.16.1%2Bcu118-cp310-cp310-linux_x86_64.whl
pip install --no-index --find-links /root/wheels torch==2.1.1 torchvision==0.16.1
python -c "import torch; print('TORCH_OK', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- OpenMMLab ----
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
mim install "mmdet==3.2.0"

# ---- mmdet3d v1.4.0 ----
cd /root
git clone --depth 1 --branch v1.4.0 https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e .

# ---- spconv + 依赖锁定 ----
pip install cumm-cu118 spconv-cu118
pip install nuscenes-devkit numpy==1.23.5 yapf==0.40.1 setuptools==58.0.4

# ---- 编译 BEVFusion 算子 ----
python projects/BEVFusion/setup.py develop

# ---- 官方权重 ----
mkdir -p checkpoints
wget -c -q https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth -P checkpoints/
wget -c -q https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth -P checkpoints/
md5sum checkpoints/*.pth

# ---- 最终验证 ----
python -c "import mmdet3d, spconv, torch; print('IMPORT_OK mmdet3d', mmdet3d.__version__)"
ls -la checkpoints/
echo "SETUP_ALL_DONE"
