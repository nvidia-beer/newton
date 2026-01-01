# Base stage
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

# Install Python 3.11 and essential tools in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install PyTorch with CUDA support
RUN pip install --index-url https://download.pytorch.org/whl/cu130 torch

# Install uv (faster Python package manager) and Jupyter/notebook dependencies
RUN python -m pip install --no-cache-dir \
    uv \
    ipykernel \
    tqdm \
    ipywidgets \
    matplotlib \
    rerun-sdk \
    rerun-notebook

# for xeyes visualization checking dcv 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        x11-apps \
        xauth \
        && rm -rf /var/lib/apt/lists/*



# Install OpenGL libraries and GPU drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    mesa-utils \
    libglfw3 \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set NVIDIA visible devices
ENV NVIDIA_VISIBLE_DEVICES=all
# Set NVIDIA driver capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=all
# Set MuJoCo rendering backend
ENV MUJOCO_GL=glfw
# Disable CUDA-OpenGL interop to prevent warnings in containerized environments
ENV NEWTON_DISABLE_CUDA_INTEROP=1
# Additional environment variables for better GPU context
ENV __GL_SYNC_TO_VBLANK=0
ENV __GL_SYNC_DISPLAY_DEVICE=DP-0
ENV LIBGL_ALWAYS_SOFTWARE=0
