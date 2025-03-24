ARG UBUNTU_VER=22.04
ARG FORGE_VER=24.11.3-2
ARG OS_TYPE=x86_64
ARG PYTHON_VERSION=3.9


FROM ubuntu:${UBUNTU_VER} 
ENV VENV_NAME=sapien-py39

# Use bash as a default shell.
SHELL ["/bin/bash", "-c"]

ENV PIP_ROOT_USER_ACTION=ignore

# System packages 
RUN apt-get update && apt-get install -y --no-install-recommends vim jq tmux bzip2 wget ssh unzip git \
    iproute2 iputils-ping build-essential curl cmake ca-certificates libglib2.0-0 libxext6 libsm6 \
    libxrender1 libpoco-dev libeigen3-dev libspdlog-dev libopenblas-dev libxcursor-dev libxrandr-dev \
    libxinerama-dev libxi-dev mesa-common-dev make gcc-10 g++-10 vulkan-tools mesa-vulkan-drivers pigz libegl1

ARG FORGE_VER
ARG OS_TYPE

# Install Miniforge3
RUN wget "https://github.com/conda-forge/miniforge/releases/download/${FORGE_VER}/Miniforge3-${FORGE_VER}-Linux-${OS_TYPE}.sh" -O miniforge.sh && \
    bash miniforge.sh -b -p /miniforge3 && \
    rm miniforge.sh
ENV PATH=/miniforge3/bin:${PATH}
RUN echo "source /miniforge3/etc/profile.d/conda.sh"  >> ~/.bashrc
RUN echo "source /miniforge3/etc/profile.d/mamba.sh"  >> ~/.bashrc


RUN mamba create -n ${VENV_NAME} python=3.9 && echo "mamba activate ${VENV_NAME}" >> ~/.bashrc
