#!/bin/bash

# gRINN Docker Build Script
# This script builds gRINN Docker images with different GROMACS versions

set -e  # Exit on any error

# Default values
GROMACS_VERSION="2024.1"
IMAGE_TAG="grinn"
DOCKERFILE="Dockerfile"
BUILD_DEV=false
PLATFORM=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build gRINN Docker images with different GROMACS versions"
    echo ""
    echo "Options:"
    echo "  -v, --version GROMACS_VERSION    GROMACS version to build (default: 2024.1)"
    echo "  -t, --tag IMAGE_TAG              Docker image tag (default: grinn)"
    echo "  -d, --dev                        Build development version (Dockerfile.dev)"
    echo "  -p, --platform PLATFORM         Build for specific platform (e.g., linux/amd64, linux/arm64)"
    echo "  -l, --list                       List available GROMACS versions"
    echo "  -h, --help                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                     # Build with GROMACS 2024.1"
    echo "  $0 -v 2023.3                          # Build with GROMACS 2023.3"
    echo "  $0 -v 2024.1 -t grinn:latest          # Build with custom tag"
    echo "  $0 -d -v 2023.3                       # Build dev version with GROMACS 2023.3"
    echo "  $0 -p linux/arm64 -v 2024.1           # Build for ARM64 (Apple Silicon)"
    echo ""
    echo "Supported GROMACS versions:"
    echo "  2020.6, 2021.6, 2022.5, 2023.3, 2024.1, 2024.2"
}

# Function to list supported GROMACS versions
list_versions() {
    echo -e "${BLUE}Supported GROMACS versions:${NC}"
    echo "  2020.6 - Stable release (older)"
    echo "  2021.6 - Stable release"
    echo "  2022.5 - Stable release"
    echo "  2023.3 - Stable release"
    echo "  2024.1 - Latest stable (default)"
    echo "  2024.2 - Latest development"
    echo ""
    echo "Note: Older versions may have compatibility issues with newer systems."
}

# Function to validate GROMACS version
validate_version() {
    local version=$1
    local supported_versions=("2020.6" "2021.6" "2022.5" "2023.3" "2024.1" "2024.2")
    
    for supported in "${supported_versions[@]}"; do
        if [[ "$version" == "$supported" ]]; then
            return 0
        fi
    done
    
    echo -e "${RED}Error: Unsupported GROMACS version '$version'${NC}"
    echo -e "Use ${YELLOW}$0 --list${NC} to see supported versions"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            GROMACS_VERSION="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -d|--dev)
            BUILD_DEV=true
            DOCKERFILE="Dockerfile.dev"
            shift
            ;;
        -p|--platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        -l|--list)
            list_versions
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate inputs
validate_version "$GROMACS_VERSION"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    echo -e "${RED}Error: $DOCKERFILE not found${NC}"
    exit 1
fi

# Build the image
echo -e "${BLUE}Building gRINN Docker image...${NC}"
echo -e "${YELLOW}GROMACS Version:${NC} $GROMACS_VERSION"
echo -e "${YELLOW}Image Tag:${NC} $IMAGE_TAG"
echo -e "${YELLOW}Dockerfile:${NC} $DOCKERFILE"
if [[ -n "$PLATFORM" ]]; then
    echo -e "${YELLOW}Platform:${NC} ${PLATFORM#--platform }"
fi
echo ""

# Construct the final image tag
if [[ "$BUILD_DEV" == true ]]; then
    FINAL_TAG="${IMAGE_TAG}:gromacs-${GROMACS_VERSION}-dev"
else
    FINAL_TAG="${IMAGE_TAG}:gromacs-${GROMACS_VERSION}"
fi

# Build command
BUILD_CMD="docker build $PLATFORM --build-arg GROMACS_VERSION=$GROMACS_VERSION -f $DOCKERFILE -t $FINAL_TAG ."

echo -e "${BLUE}Executing:${NC} $BUILD_CMD"
echo ""

# Start timing
start_time=$(date +%s)

# Execute the build
if eval "$BUILD_CMD"; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo ""
    echo -e "${GREEN}✓ Build completed successfully!${NC}"
    echo -e "${GREEN}✓ Image built:${NC} $FINAL_TAG"
    echo -e "${GREEN}✓ Build time:${NC} ${duration} seconds"
    echo ""
    echo -e "${BLUE}Usage examples:${NC}"
    echo "  # Run workflow"
    echo "  docker run -v /data:/data $FINAL_TAG workflow /data/protein.pdb /data/results"
    echo ""
    echo "  # Run dashboard"
    echo "  docker run -p 8051:8051 -v /data:/data $FINAL_TAG dashboard /data/results"
    echo ""
    echo "  # Interactive bash"
    echo "  docker run -it $FINAL_TAG bash"
    echo ""
    
    # Also tag as latest if this is the default version
    if [[ "$GROMACS_VERSION" == "2024.1" ]] && [[ "$BUILD_DEV" == false ]]; then
        docker tag "$FINAL_TAG" "${IMAGE_TAG}:latest"
        echo -e "${GREEN}✓ Also tagged as:${NC} ${IMAGE_TAG}:latest"
    fi
    
else
    echo ""
    echo -e "${RED}✗ Build failed!${NC}"
    echo -e "${YELLOW}Common issues:${NC}"
    echo "  - Network connectivity problems"
    echo "  - Insufficient disk space"
    echo "  - Platform compatibility issues"
    echo ""
    echo "For ARM64 (Apple Silicon), try:"
    echo "  $0 --platform linux/arm64 --version $GROMACS_VERSION"
    exit 1
fi
