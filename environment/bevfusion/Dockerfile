FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3.8-venv build-essential git curl ca-certificates \
    libgl1 libglib2.0-0 libopenmpi-dev openmpi-bin openssh-server && \
    rm -rf /var/lib/apt/lists/*
RUN python3.8 -m venv /opt/venvs/bevfusion
ENV PATH=/opt/venvs/bevfusion/bin:$PATH CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" MAX_JOBS=4
COPY environment/bevfusion /opt/deployment/bevfusion
RUN bash /opt/deployment/bevfusion/setup.sh
WORKDIR /opt/bevfusion
CMD ["sleep", "infinity"]
