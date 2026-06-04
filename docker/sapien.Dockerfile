FROM ubuntu:24.04

ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv
ENV VIRTUAL_ENV=/workspace/.venv
ENV UV_PYTHON=${PYTHON_VERSION}
ENV PATH="/workspace/.venv/bin:/root/.local/bin:${PATH}"
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    iproute2 \
    iputils-ping \
    jq \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libegl1 \
    libeigen3-dev \
    libglib2.0-0 \
    libopenblas-dev \
    libpoco-dev \
    libsm6 \
    libspdlog-dev \
    libusb-1.0-0-dev \
    libxcursor-dev \
    libxext6 \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxrender1 \
    make \
    mesa-common-dev \
    mesa-vulkan-drivers \
    pigz \
    ssh \
    tmux \
    unzip \
    vim \
    vulkan-tools \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

COPY pyproject.toml uv.lock .python-version ./
ENV UV_LINK_MODE=copy
RUN uv sync --locked --no-install-project

RUN mkdir -p /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
COPY docker/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN curl -L -o physxgpu-linux-clang.zip \
    https://github.com/yolkarian/physx-release/releases/download/107.3-physx-5.6.1-Linux/physxgpu-linux-clang.zip \
    && mkdir -p /root/.sapien/physx/107.3-physx-5.6.1 \
    && unzip physxgpu-linux-clang.zip -d /root/.sapien/physx/107.3-physx-5.6.1 \
    && rm physxgpu-linux-clang.zip

COPY docker/entrypoint.sh /usr/local/bin/furniture-bench-entrypoint
RUN chmod +x /usr/local/bin/furniture-bench-entrypoint

ENTRYPOINT ["/usr/local/bin/furniture-bench-entrypoint"]
CMD ["bash"]