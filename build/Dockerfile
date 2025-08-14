# syntax=docker/dockerfile:1.7

# Build args so you can switch CUDA builds at build time
ARG CUDA_VERSION=12.4

# Base image already includes torch==2.6.0 for the right CUDA
FROM pytorch/pytorch:2.6.0-cuda${CUDA_VERSION}-cudnn9-runtime

ARG CU_VERSION="124"

# Avoid interactive tzdata etc.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Copy only files that affect dependency resolution to leverage Docker cache
COPY pyproject.toml README.md ./

# 2) Install base + container 
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir .[container]

# 3) Install `torch-cluster`
RUN python -m pip install --no-cache-dir \
    -f https://data.pyg.org/whl/torch-2.6.0+cu${CU_VERSION}.html \
    torch-cluster

# Default command
CMD ["python", "--version"]

# Example build command
# docker build --build-arg CUDA_VERSION=12.4 --build-arg CU_VERSION=124 -t dronalize:latest -f build/Dockerfile .
