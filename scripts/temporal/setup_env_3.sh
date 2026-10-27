#!/bin/bash
# 第 3 段:改走 cu121(与 nvcc 12.8 主版本一致),补齐 nvidia 依赖,重装 mmdet3d + 编译 BEVFusion
exec > /root/autodl-tmp/setup3.log 2>&1
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bevfusion

BASE=https://download.pytorch.org/whl/cu121
cd /root/wheels
rm -f torch-2.1.1+cu118*.whl torchvision-0.16.1+cu118*.whl

# ---- torch / torchvision cu121 ----
wget -c -q $BASE/torch-2.1.1%2Bcu121-cp310-cp310-linux_x86_64.whl
wget -c -q $BASE/torchvision-0.16.1%2Bcu121-cp310-cp310-linux_x86_64.whl

# ---- 按 torch METADATA 精确下载 nvidia 依赖 ----
unzip -p torch-2.1.1+cu121-cp310-cp310-linux_x86_64.whl torch-2.1.1+cu121.dist-info/METADATA \
  | grep -oE 'nvidia-[a-z0-9-]+==[0-9.]+' | tr '==' '  ' | while read pkg ver; do
    fname=$(curl -s $BASE/$pkg/ | grep -oE "$(echo $pkg | tr '-' '_')-${ver}[^\"<>]*\.whl" | head -1)
    echo "FETCH_NVIDIA $pkg $ver $fname"
    [ -n "$fname" ] && wget -c -q $BASE/$pkg/$fname || echo "MISSING $pkg"
  done

# ---- triton(torch 2.1.1 linux 需要 pytorch-triton==2.1.0)----
tfname=$(curl -s $BASE/pytorch-triton/ | grep -oE 'pytorch_triton-2\.1\.0[^\"<>]*\.whl' | head -1)
echo "FETCH_TRITON $tfname"
[ -n "$tfname" ] && wget -c -q $BASE/pytorch-triton/$tfname \
  || pip download pytorch-triton==2.1.0 --no-deps -d /root/wheels

# ---- 其余小型纯 python 依赖走阿里云镜像 ----
pip download sympy networkx jinja2 fsspec --no-deps -d /root/wheels

# ---- 安装 torch ----
pip install --no-index --find-links /root/wheels torch==2.1.1 torchvision==0.16.1
python -c "import torch; print('TORCH_OK', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- 换 cu121 版 spconv ----
pip uninstall -y cumm-cu118 spconv-cu118
pip install cumm-cu121 spconv-cu121

# ---- OpenMMLab 三件套(mim 会自动识别 torch2.1+cu121)----
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"

# ---- mmdet3d 可编辑安装(必须 --no-build-isolation,否则构建环境里没有 torch)----
cd /root/mmdetection3d
pip install -v -e . --no-build-isolation
python -c "import mmdet3d; print('MMDET3D_OK', mmdet3d.__version__)"

# ---- 锁 numpy(防 numpy2 兼容问题)----
pip install "numpy==1.23.5"

# ---- 编译 BEVFusion 算子(4090D = sm_89)----
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
python projects/BEVFusion/setup.py develop

# ---- 最终自检 ----
python -c "
import torch, mmdet3d
print('IMPORT_OK torch', torch.__version__, 'cuda', torch.version.cuda, '| mmdet3d', mmdet3d.__version__)
import mmcv, mmengine, mmdet
print('MMCV', mmcv.__version__, 'MMENGINE', mmengine.__version__, 'MMDET', mmdet.__version__)
x = torch.randn(4, 4).cuda(); print('CUDA_MATMUL_OK', (x @ x).sum().item() != 0)
"
echo "SETUP3_ALL_DONE"
