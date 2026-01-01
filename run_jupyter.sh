#!/bin/bash
# Build and run Docker container (like devcontainer) with Jupyter for remote browser access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${JUPYTER_PORT:-8888}

echo "=============================================="
echo "  Building & Running Docker + Jupyter"
echo "=============================================="

# Build the Docker image from .devcontainer/Dockerfile
echo "Building Docker image..."
docker build -t newton-jupyter "${SCRIPT_DIR}/.devcontainer"

echo ""
echo "Starting Docker container with Jupyter..."
echo ""

# Run the container with the same settings as devcontainer.json
# Then run Jupyter inside it
docker run -it --rm \
    --name newton-jupyter \
    --network=host \
    --ipc=host \
    --gpus=all \
    --shm-size=16g \
    --privileged \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e DISPLAY="${DISPLAY}" \
    -e XAUTHORITY=/root/.Xauthority \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NEWTON_DISABLE_CUDA_INTEROP=1 \
    -e __GL_SYNC_TO_VBLANK=0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /run/user/1000/dcv/session.xauth:/root/.Xauthority:ro \
    -v "${SCRIPT_DIR}:/workspaces/newton" \
    -w /workspaces/newton \
    newton-jupyter \
    bash -c '
        echo "Setting up Python environment..."
        
        # Install uv if not available
        if ! command -v uv &> /dev/null; then
            pip install uv
        fi
        
        cd /workspaces/newton
        
        # Sync dependencies (this installs newton in the venv)
        uv sync --extra sim --extra examples --dev || true
        
        # Install jupyter and ipykernel in the venv using uv pip
        uv pip install ipykernel tqdm ipywidgets matplotlib rerun-sdk rerun-notebook jupyter jupyterlab
        
        # Register the venv as a Jupyter kernel
        uv run python -m ipykernel install --user --name newton --display-name "Python (newton)"
        
        echo ""
        echo "=============================================="
        echo "  Jupyter Lab Starting..."
        echo "  Access at: http://'"$(curl -s ifconfig.me)"':'"${PORT}"'"
        echo "  Tutorial notebooks in: /workspaces/newton/tutorial"
        echo ""
        echo "  IMPORTANT: Select kernel \"Python (newton)\" in Jupyter!"
        echo "=============================================="
        echo ""
        
        # Run Jupyter Lab from the venv using uv run
        uv run jupyter lab \
            --ip=0.0.0.0 \
            --port='"${PORT}"' \
            --no-browser \
            --allow-root \
            --ServerApp.token="" \
            --ServerApp.password="" \
            --ServerApp.allow_origin="*" \
            --ServerApp.allow_remote_access=True \
            --notebook-dir=/workspaces/newton/tutorial
    '
