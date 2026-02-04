# Newton Docker Setup

This directory contains Docker configurations for running Newton standalone (without Isaac Lab).

## Quick Start

### Build Docker Image

```bash
# Auto-detect platform and build
./.devcontainer/build-docker.sh

# Or specify platform
./.devcontainer/build-docker.sh arm64  # For ARM64/aarch64
./.devcontainer/build-docker.sh x86    # For x86_64/amd64
./.devcontainer/build-docker.sh both   # Build both architectures
```

### Run Examples

```bash
# Interactive menu
./.devcontainer/run-examples.sh

# Run specific example
./.devcontainer/run-examples.sh bouncing_ball
./.devcontainer/run-examples.sh rigid_soft_interaction --solver xpbd
./.devcontainer/run-examples.sh inflatable_rigid --solver mujoco
```

## Files

- **`Dockerfile.arm64`** - ARM64 build using NVIDIA CUDA 13 base (matches warp-lang v1.11.0+cu13)
- **`Dockerfile.x86`** - x86_64 build using UV Python 3.11 base
- **`build-docker.sh`** - Build script with platform auto-detection
- **`run-examples.sh`** - Run Newton examples in Docker with interactive menus

## Architecture Notes

### ARM64 Build
- Base: `nvidia/cuda:13.0.0-devel-ubuntu22.04`
- Matches warp-lang v1.11.0+cu13 CUDA requirements
- GPU-accelerated OpenGL via GLVND (vendor-neutral dispatch)

### x86_64 Build
- Base: `ghcr.io/astral-sh/uv:python3.11-bookworm`
- Fast, reproducible builds with UV package manager
- Mesa OpenGL software rendering (GPU acceleration via NVIDIA runtime)

## Requirements

- Docker with BuildKit support
- NVIDIA GPU + drivers (for GPU acceleration)
- X11 display (for viewer)

## GPU Support

The run script automatically detects NVIDIA GPUs:
- ✓ GPU detected → Enables GPU acceleration
- ⚠ No GPU → Falls back to CPU mode (slower but functional)
