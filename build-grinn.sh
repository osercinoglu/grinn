#!/bin/bash

# Unified gRINN Build Script
# Builds gRINN Docker images with any supported GROMACS version
# 
# Usage:
#   ./build-grinn.sh <GROMACS_VERSION> [OPTIONS]
# 
# Examples:
#   ./build-grinn.sh 2024.1
#   ./build-grinn.sh 2025.2
#   ./build-grinn.sh 2020.7
#   ./build-grinn.sh 2023.4 --no-cache

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

show_usage() {
    echo "gRINN Docker Build Script"
    echo ""
    echo "Usage: $0 <GROMACS_VERSION> [OPTIONS]"
    echo ""
    echo "Supported GROMACS Versions:"
    echo "  2020.7    - Legacy version (Ubuntu 20.04)"
    echo "  2021.7    - Ubuntu 22.04"
    echo "  2022.x    - Ubuntu 22.04"
    echo "  2023.x    - Ubuntu 22.04"
    echo "  2024.x    - Ubuntu 22.04"
    echo "  2025.x    - Ubuntu 22.04 + Modern CMake"
    echo ""
    echo "Options:"
    echo "  --no-cache     Build without using Docker cache"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 2024.1"
    echo "  $0 2025.2 --no-cache"
    echo "  $0 2020.7"
    echo ""
}

# Parse arguments
if [ $# -eq 0 ]; then
    print_error "No GROMACS version specified"
    show_usage
    exit 1
fi

GROMACS_VERSION="$1"
shift

# Parse options
NO_CACHE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Find the grinn repository root directory
find_grinn_root() {
    local current_dir="$(pwd)"
    
    # Check if we're in the grinn directory or a parent directory containing it
    while [[ "$current_dir" != "/" ]]; do
        if [[ -f "$current_dir/grinn_workflow.py" && -d "$current_dir/docker-images" ]]; then
            echo "$current_dir"
            return 0
        fi
        
        if [[ -f "$current_dir/grinn/grinn_workflow.py" && -d "$current_dir/grinn/docker-images" ]]; then
            echo "$current_dir"
            return 0
        fi
        
        current_dir="$(dirname "$current_dir")"
    done
    
    print_error "Could not find grinn repository root"
    print_info "Please run this script from within the grinn repository"
    exit 1
}

# Validate GROMACS version format
validate_version() {
    if [[ ! "$GROMACS_VERSION" =~ ^[0-9]{4}\.[0-9]+$ ]]; then
        print_error "Invalid GROMACS version format: $GROMACS_VERSION"
        print_info "Expected format: YYYY.X (e.g., 2024.1, 2025.2)"
        exit 1
    fi
}

# Determine build context and file prefix
setup_build_context() {
    GRINN_ROOT="$(find_grinn_root)"
    
    if [[ -f "$GRINN_ROOT/grinn/grinn_workflow.py" ]]; then
        # grinn is a subdirectory
        BUILD_CONTEXT="$GRINN_ROOT"
        GRINN_PREFIX="grinn/"
        print_info "Build context: $BUILD_CONTEXT (grinn as subdirectory)"
    elif [[ -f "$GRINN_ROOT/grinn_workflow.py" ]]; then
        # We're in the grinn directory itself
        BUILD_CONTEXT="$GRINN_ROOT"
        GRINN_PREFIX=""
        print_info "Build context: $BUILD_CONTEXT (grinn root directory)"
    else
        print_error "Could not find grinn_workflow.py in expected locations"
        exit 1
    fi
}

# Select appropriate Dockerfile
select_dockerfile() {
    if [[ "$GROMACS_VERSION" == "2020.7" ]]; then
        DOCKERFILE_PATH="$BUILD_CONTEXT/${GRINN_PREFIX}Dockerfile.gromacs-2020.7"
        IMAGE_TAG="grinn:gromacs-2020.7"
        DESCRIPTION="Legacy GROMACS 2020.7 (Ubuntu 20.04)"
    else
        DOCKERFILE_PATH="$BUILD_CONTEXT/${GRINN_PREFIX}Dockerfile.unified"
        IMAGE_TAG="grinn:gromacs-$GROMACS_VERSION"
        DESCRIPTION="Modern GROMACS $GROMACS_VERSION (Ubuntu 22.04)"
    fi
    
    if [[ ! -f "$DOCKERFILE_PATH" ]]; then
        print_error "Dockerfile not found: $DOCKERFILE_PATH"
        exit 1
    fi
}

# Estimate build time
estimate_build_time() {
    local gromacs_major=$(echo "$GROMACS_VERSION" | cut -d. -f1)
    
    if [[ "$GROMACS_VERSION" == "2020.7" ]]; then
        ESTIMATED_TIME="15-20 minutes"
    elif [[ "$gromacs_major" -ge "2025" ]]; then
        ESTIMATED_TIME="25-30 minutes"
    else
        ESTIMATED_TIME="20-25 minutes"
    fi
}

# Main build function
build_image() {
    print_info "================================================================"
    print_info "Building gRINN with $DESCRIPTION"
    print_info "================================================================"
    print_info "GROMACS Version: $GROMACS_VERSION"
    print_info "Dockerfile: $DOCKERFILE_PATH"
    print_info "Build Context: $BUILD_CONTEXT"
    print_info "File Prefix: '$GRINN_PREFIX'"
    print_info "Image Tag: $IMAGE_TAG"
    print_info "Estimated Time: $ESTIMATED_TIME"
    print_info ""
    
    # Build command
    BUILD_CMD="docker build"
    
    if [[ -n "$NO_CACHE" ]]; then
        BUILD_CMD="$BUILD_CMD $NO_CACHE"
        print_warning "Building without cache (will take longer)"
    fi
    
    if [[ "$GROMACS_VERSION" != "2020.7" ]]; then
        BUILD_CMD="$BUILD_CMD --build-arg GROMACS_VERSION=$GROMACS_VERSION"
    fi
    
    BUILD_CMD="$BUILD_CMD --build-arg GRINN_PREFIX=$GRINN_PREFIX"
    BUILD_CMD="$BUILD_CMD -f $DOCKERFILE_PATH"
    BUILD_CMD="$BUILD_CMD -t $IMAGE_TAG"
    BUILD_CMD="$BUILD_CMD $BUILD_CONTEXT"
    
    print_info "Build command: $BUILD_CMD"
    print_info ""
    
    # Execute build
    if eval "$BUILD_CMD"; then
        print_success "Build completed successfully!"
    else
        print_error "Build failed!"
        exit 1
    fi
}

# Verify the built image
verify_image() {
    print_info "================================================================"
    print_info "Verifying GROMACS $GROMACS_VERSION installation..."
    print_info "================================================================"
    
    if docker run --rm "$IMAGE_TAG" gmx --version | grep -q "$GROMACS_VERSION"; then
        print_success "GROMACS $GROMACS_VERSION installed correctly"
    else
        print_error "GROMACS $GROMACS_VERSION verification failed"
        exit 1
    fi
    
    if docker run --rm "$IMAGE_TAG" workflow --help > /dev/null 2>&1; then
        print_success "gRINN workflow accessible"
    else
        print_error "gRINN workflow verification failed"
        exit 1
    fi
}

# Show final information
show_summary() {
    print_info "================================================================"
    print_success "BUILD COMPLETED SUCCESSFULLY!"
    print_info "================================================================"
    print_info "Image: $IMAGE_TAG"
    print_info "GROMACS Version: $GROMACS_VERSION"
    print_info "Description: $DESCRIPTION"
    print_info ""
    print_info "Usage Examples:"
    print_info "  docker run --rm $IMAGE_TAG gmx --version"
    print_info "  docker run --rm $IMAGE_TAG workflow --help"
    print_info "  docker run -p 8051:8051 -v /data:/data $IMAGE_TAG dashboard /data/results"
    print_info "  docker run -it $IMAGE_TAG bash"
    print_info ""
    print_info "Next steps:"
    print_info "  1. Test the image with your data"
    print_info "  2. Push to registry if needed"
    print_info "  3. Build additional GROMACS versions"
    print_info ""
}

# Main execution
main() {
    validate_version
    setup_build_context
    select_dockerfile
    estimate_build_time
    build_image
    verify_image
    show_summary
}

# Run main function
main
