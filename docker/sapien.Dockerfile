FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ARG FORGE_VER=24.11.3-2
ARG OS_TYPE=x86_64
ARG PYTHON_VERSION=3.11


ENV VENV_NAME=furniture-bench-gpu

# Use bash as a default shell.
SHELL ["/bin/bash", "-c"]

ENV PIP_ROOT_USER_ACTION=ignore

# System packages 
RUN apt-get update && apt-get install -y --no-install-recommends vim jq tmux bzip2 wget ssh unzip git \
    iproute2 iputils-ping build-essential curl cmake ca-certificates libglib2.0-0 libxext6 libsm6 \
    libxrender1 libpoco-dev libeigen3-dev libspdlog-dev libopenblas-dev libxcursor-dev libxrandr-dev \
    libxinerama-dev libxi-dev mesa-common-dev make gcc-10 g++-10 vulkan-tools mesa-vulkan-drivers pigz libegl1 \
    && apt install -y --no-install-recommends libcanberra-gtk-module libcanberra-gtk3-module libusb-1.0-0-dev 

ARG FORGE_VER
ARG OS_TYPE

# Install Miniforge3
RUN wget "https://github.com/conda-forge/miniforge/releases/download/${FORGE_VER}/Miniforge3-${FORGE_VER}-Linux-${OS_TYPE}.sh" -O miniforge.sh && \
    bash miniforge.sh -b -p /miniforge3 && \
    rm miniforge.sh
ENV PATH=/miniforge3/bin:${PATH}
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# Config Conda and Mamba
RUN echo "source /miniforge3/etc/profile.d/conda.sh"  >> ~/.bashrc && \ 
    echo "source /miniforge3/etc/profile.d/mamba.sh"  >> ~/.bashrc

# Create Env with Mamba
COPY environment.yml /tmp/environment.yml
RUN CONDA_OVERRIDE_CUDA="12.9" mamba env create -y -n ${VENV_NAME} -f /tmp/environment.yml && \
    echo "mamba activate ${VENV_NAME}" >> ~/.bashrc && \
    mamba clean -a -y && \
    rm /tmp/environment.yml

RUN mamba run -n ${VENV_NAME} pip install https://github.com/yolkarian/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20260326+4b2eaf21-cp311-cp311-manylinux_2_28_x86_64.whl && \
    wget https://github.com/yolkarian/physx-release/releases/download/107.3-physx-5.6.1-Linux/physxgpu-linux-clang.zip && \
    mkdir -p /root/.sapien/physx/107.3-physx-5.6.1 && \
    unzip physxgpu-linux-clang.zip -d /root/.sapien/physx/107.3-physx-5.6.1 && \
    rm physxgpu-linux-clang.zip

# Install the project in editable mode (source copied last for layer caching)
WORKDIR /root/furniture-bench
COPY . .
RUN mamba run -n ${VENV_NAME} pip install --no-deps -e .
