#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# docker/ lives at <repo>/docker; build context is the repo root (one level up)
NEWTON_REPO_PATH="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [PLATFORM]"
    echo ""
    echo "PLATFORM can be:"
    echo "  auto     - Auto-detect platform (default)"
    echo "  arm64    - Build for ARM64/aarch64"
    echo "  x86      - Build for x86_64/amd64"
    echo "  both     - Build for both platforms (creates multi-arch image)"
    echo ""
    echo "Examples:"
    echo "  $0           # Auto-detect and build for current platform"
    echo "  $0 arm64     # Force build for ARM64"
    echo "  $0 both      # Build multi-arch image for both platforms"
    exit 1
}

PLATFORM="${1:-auto}"

if [ "$PLATFORM" = "auto" ]; then
    ARCH=$(uname -m)
    case "$ARCH" in
        aarch64|arm64)
            PLATFORM="arm64"
            ;;
        x86_64|amd64)
            PLATFORM="x86"
            ;;
        *)
            echo -e "${YELLOW}Warning: Unknown architecture '$ARCH', defaulting to x86${NC}"
            PLATFORM="x86"
            ;;
    esac
    echo -e "${BLUE}Auto-detected platform: $PLATFORM${NC}"
    echo ""
fi

case "$PLATFORM" in
    arm64)
        ARCH_TAG="arm64"
        echo -e "${BLUE}Building Newton for ARM64 (aarch64)...${NC}"
        echo "Using NVIDIA CUDA 13 base image"
        echo ""

        DOCKER_BUILDKIT=1 docker build \
            --platform linux/${ARCH_TAG} \
            -t newton:${ARCH_TAG} \
            -t newton:latest \
            -f "$SCRIPT_DIR/Dockerfile.arm64" \
            "$NEWTON_REPO_PATH"

        echo ""
        echo -e "${GREEN}✓ ARM64 build complete!${NC}"
        echo "Tagged as: newton:${ARCH_TAG}, newton:latest"
        echo "Run with: docker run --rm -it --gpus all newton:latest"
        ;;

    x86)
        ARCH_TAG="amd64"
        echo -e "${BLUE}Building Newton for x86_64 (amd64)...${NC}"
        echo "Using UV Python 3.11 base image"
        echo ""

        DOCKER_BUILDKIT=1 docker build \
            --platform linux/${ARCH_TAG} \
            -t newton:${ARCH_TAG} \
            -t newton:latest \
            -f "$SCRIPT_DIR/Dockerfile.x86" \
            "$NEWTON_REPO_PATH"

        echo ""
        echo -e "${GREEN}✓ x86_64 build complete!${NC}"
        echo "Tagged as: newton:${ARCH_TAG}, newton:latest"
        echo "Run with: docker run --rm -it --gpus all newton:latest"
        ;;

    both)
        echo -e "${BLUE}Building Newton for multiple architectures...${NC}"
        echo ""
        echo -e "${YELLOW}Note: This requires Docker buildx and may take significant time.${NC}"
        echo ""

        if ! docker buildx version &> /dev/null; then
            echo -e "${YELLOW}Error: docker buildx is not available${NC}"
            echo "Please install Docker Buildx to build multi-architecture images"
            exit 1
        fi

        if ! docker buildx inspect multiarch-builder &> /dev/null; then
            echo "Creating buildx builder instance..."
            docker buildx create --name multiarch-builder --use
        else
            docker buildx use multiarch-builder
        fi

        echo -e "${BLUE}Building ARM64 image...${NC}"
        DOCKER_BUILDKIT=1 docker buildx build \
            --platform linux/arm64 \
            -t newton:arm64 \
            -f "$SCRIPT_DIR/Dockerfile.arm64" \
            --load \
            "$NEWTON_REPO_PATH"

        echo ""
        echo -e "${BLUE}Building x86_64 image...${NC}"
        DOCKER_BUILDKIT=1 docker buildx build \
            --platform linux/amd64 \
            -t newton:amd64 \
            -f "$SCRIPT_DIR/Dockerfile.x86" \
            --load \
            "$NEWTON_REPO_PATH"

        CURRENT_ARCH=$(uname -m)
        case "$CURRENT_ARCH" in
            aarch64|arm64)
                docker tag newton:arm64 newton:latest
                echo -e "${GREEN}Tagged newton:arm64 as latest (matching current platform)${NC}"
                ;;
            x86_64|amd64)
                docker tag newton:amd64 newton:latest
                echo -e "${GREEN}Tagged newton:amd64 as latest (matching current platform)${NC}"
                ;;
        esac

        echo ""
        echo -e "${GREEN}✓ Multi-architecture build complete!${NC}"
        echo "Available images:"
        echo "  - newton:arm64  (for ARM64/aarch64)"
        echo "  - newton:amd64  (for x86_64/amd64)"
        echo "  - newton:latest (matches your current platform)"
        echo ""
        echo "Run with: docker run --rm -it --gpus all newton:latest"
        ;;

    *)
        echo -e "${YELLOW}Error: Unknown platform '$PLATFORM'${NC}"
        echo ""
        usage
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
