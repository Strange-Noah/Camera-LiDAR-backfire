FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    g++ \
    gcc \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python -m pip install --upgrade pip "setuptools<81" wheel

# Install PyTorch with CUDA 12.8 support for RTX 50-series
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
RUN pip install Pillow tqdm torchpack openmim nuscenes-devkit mpi4py numba

# MMCV 2.1 imports pkg_resources during source builds. Setuptools 81+ removed it.
RUN printf "setuptools<81\n" > /tmp/pip-constraints.txt
ENV PIP_CONSTRAINT=/tmp/pip-constraints.txt

ENV MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2

RUN mim install "mmengine>=0.8.0,<1.0.0"
RUN python -m pip install --force-reinstall "setuptools<81" wheel
RUN python -m pip install --no-build-isolation -v mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.7.0/index.html
RUN pip install mmdet==3.3.0