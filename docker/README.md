# Newton Docker Setup

Docker configuration to build and run Newton standalone (without Isaac Lab).

Adapted from the `.devcontainer` setup in the legacy
`Newton/isaac-lab/newton` project, but targeting this repo's layout.

## Layout

This `docker/` folder lives at the Newton repo root
(`third_party/newton/docker/`), parallel to `pyproject.toml` / `uv.lock`.
The build context is the parent directory (the Newton repo root) so the
Dockerfiles' `COPY pyproject.toml uv.lock ./` and friends Just Work.

## Quick Start

### Build

```bash
# Auto-detect current platform
./docker/build-docker.sh

# Or pick a platform explicitly
./docker/build-docker.sh x86     # x86_64 / amd64
./docker/build-docker.sh arm64   # aarch64 (Jetson / Grace)
./docker/build-docker.sh both    # multi-arch with buildx
```

Produces an image tagged `newton:latest` (and `newton:amd64` or
`newton:arm64`).

### Run an example

```bash
# Interactive menu
./docker/run-examples.sh

# By name (flat example name; see newton/examples/*/example_*.py)
./docker/run-examples.sh basic_pendulum
./docker/run-examples.sh robot_cartpole --num-frames 500

# By number (from the menu order)
./docker/run-examples.sh 1

# Force a viewer
./docker/run-examples.sh --gl basic_pendulum      # default
./docker/run-examples.sh --usd basic_pendulum     # write output.usd
./docker/run-examples.sh --null robot_cartpole    # headless
```

Examples are invoked inside the container via
`python -m newton.examples <name>`. The live Python package is bind-mounted
from the host at `third_party/newton/newton` → `/workspace/newton/newton`,
so edits to Python sources take effect without rebuilding the image.

## Files

- **`Dockerfile.x86`** – x86_64 build on `ghcr.io/astral-sh/uv:python3.11-bookworm`.
- **`Dockerfile.arm64`** – ARM64 build on `nvidia/cuda:13.0.0-devel-ubuntu22.04`.
- **`build-docker.sh`** – Build entry point with platform auto-detection.
- **`run-examples.sh`** – Discovers examples from `config/*.json`, resolves
  CLI args via `run-example.py`, and runs the selected example in the
  `newton:latest` image.
- **`run-example.py`** – Host-side helper that lists configs and expands
  one JSON file into argparse CLI tokens (with optional interactive edit).
- **`config/`** – One JSON file per example. Each file holds a short
  description plus the example's argparse defaults (base parser + any
  args added by the example itself, pulled straight from the source).
  See `config/README.md` for the schema.

## Runtime flow

```
run-examples.sh  ──►  run-example.py list   ──►  reads config/*.json ──►  menu
                                          ▼
                      run-example.py resolve  (optional --edit / --set)
                                          ▼
              docker run … newton:latest python -m newton.examples <name> <args>
```

CLI arg precedence (last wins under argparse):

1. Values in the JSON `args` block.
2. `--set KEY=VAL` overrides passed to `run-examples.sh`.
3. Interactive edits when `-e` / `--edit` is used.
4. Raw Newton args passed after `--` on the command line.

## Requirements

- Docker with BuildKit
- NVIDIA GPU + driver (optional; CPU fallback works for most examples)
- X11 display if using the `gl` viewer

## Viewer options

Newton in this repo exposes the following viewers (see
`newton/examples/__init__.py` → `create_parser`):
`gl`, `usd`, `rerun`, `null`, `viser`.

> Note: there is **no** RTX / Vulkan viewer in this version of Newton, so
> this setup skips the Vulkan ICD plumbing that existed in the older
> `.devcontainer`. If you later need Vulkan in-container, re-add the
> `--runtime=nvidia` flag and the `/usr/share/vulkan/icd.d` mount from the
> legacy reference script.

## GPU + X11 notes

- `run-examples.sh` detects `nvidia-smi` on the host. If present, it adds
  `--gpus all` and sets `NVIDIA_DRIVER_CAPABILITIES=all`.
- `DISPLAY` is auto-detected from `/tmp/.X11-unix/`. `XAUTHORITY` is
  forwarded if set. If the viewer window won't open, try
  `xhost +local:docker` on the host.

## Dependency extras

The images install the `dev` extra, which transitively pulls in:

- `examples` – `pyglet`, `imgui_bundle`, `GitPython`, `pyyaml`, `cbor2`, `Pillow`
- `sim` – `mujoco`, `mujoco-warp`
- `importers` – USD, mesh libs (`trimesh`, `meshio`, `scipy`, …)

This mirrors the set you get locally with `uv sync --extra dev`.
