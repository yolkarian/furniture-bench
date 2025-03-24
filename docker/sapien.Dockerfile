FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ARG FORGE_VER=24.11.3-2
ARG OS_TYPE=x86_64
ARG PYTHON_VERSION=3.9


ENV VENV_NAME=furniture-sapien-gpu

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
RUN mamba create -n ${VENV_NAME} -y python=3.9 numpy=1.23.5 pytorch==2.4.1 torchvision==0.19.1 \
    torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia && \ 
    echo "mamba activate ${VENV_NAME}" >> ~/.bashrc 

RUN mamba run -n ${VENV_NAME} pip install https://github.com/yolkarian/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250319+5d6b8739-cp39-cp39-manylinux_2_28_x86_64.whl && \
    mamba run -n ${VENV_NAME} pip install https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8/pytorch3d-0.7.8+pt2.1.0cu121-cp39-cp39-linux_x86_64.whl &&\
    wget https://github.com/sapien-sim/physx-precompiled/releases/download/105.1-physx-5.3.1.patch0/linux-so.zip && \
    mkdir -p /root/.sapien/physx/105.1-physx-5.3.1.patch0 && \
    unzip linux-so.zip -d /root/.sapien/physx/105.1-physx-5.3.1.patch0 && \
    rm linux-so.zip
    
