#!/bin/bash

# Unified gRINN Build Script
# Builds gRINN Docker images with any supported GROMACS version
# 
# Usage:
#   ./build-grinn.sh <GROMACS_VERSION> [OPTIONS]
#   ./build-grinn.sh --rebuild-all [OPTIONS]
# 
# Examples:
#   ./build-grinn.sh 2024.1
#   ./build-grinn.sh 2025.2
#   ./build-grinn.sh 2020.7
#   ./build-grinn.sh 2023.4 --no-cache
#   ./build-grinn.sh --rebuild-all
#   ./build-grinn.sh --rebuild-all --no-cache

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Supported GROMACS versions (based on README.md badge: 2020.7-2025.2)
SUPPORTED_VERSIONS=(
    "2020.7"
    "2021.7"
    "2022.1" "2022.2" "2022.3" "2022.4" "2022.5" "2022.6"
    "2023.1" "2023.2" "2023.3" "2023.4" "2023.5"
    "2024.1" "2024.2" "2024.3"
    "2025.1" "2025.2"
)

# Global flags
REBUILD_ALL=false
NO_CACHE=""

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

# Function to remove existing Docker image if it exists
remove_existing_image() {
    local image_tag="$1"
    
    if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "^$image_tag$"; then
        print_warning "Removing existing image: $image_tag"
        if docker rmi "$image_tag" 2>/dev/null; then
            print_success "Successfully removed existing image: $image_tag"
        else
            print_warning "Could not remove image $image_tag (may be in use)"
        fi
    fi
}

# Function to rebuild all supported GROMACS versions
rebuild_all_versions() {
    local no_cache_arg="$1"
    local failed_builds=()
    local successful_builds=()
    
    print_info "================================================================"
    print_info "REBUILDING ALL SUPPORTED GROMACS VERSIONS"
    print_info "================================================================"
    print_info "Supported versions: ${SUPPORTED_VERSIONS[*]}"
    print_info "Total versions to build: ${#SUPPORTED_VERSIONS[@]}"
    
    if [[ -n "$no_cache_arg" ]]; then
        print_warning "Building ALL versions without cache (this will take a very long time!)"
    fi
    
    print_info ""
    
    # Confirm with user before proceeding
    read -p "This will rebuild ${#SUPPORTED_VERSIONS[@]} Docker images. Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cancelled by user."
        exit 0
    fi
    
    local start_time=$(date +%s)
    local version_count=0
    
    for version in "${SUPPORTED_VERSIONS[@]}"; do
        version_count=$((version_count + 1))
        
        print_info ""
        print_info "================================================================"
        print_info "Building version $version_count/${#SUPPORTED_VERSIONS[@]}: GROMACS $version"
        print_info "================================================================"
        
        # Export variables for the build functions to use
        export GROMACS_VERSION="$version"
        export NO_CACHE="$no_cache_arg"
        
        # Setup build context and dockerfile selection
        setup_build_context
        select_dockerfile
        estimate_build_time
        
        # Remove existing image before building
        remove_existing_image "$IMAGE_TAG"
        
        print_info "Starting build for GROMACS $version..."
        print_info "Estimated time: $ESTIMATED_TIME"
        
        # Build the image
        if build_image_silent; then
            if verify_image_silent; then
                successful_builds+=("$version")
                print_success "✅ GROMACS $version: BUILD SUCCESSFUL"
            else
                failed_builds+=("$version")
                print_error "❌ GROMACS $version: VERIFICATION FAILED"
            fi
        else
            failed_builds+=("$version")
            print_error "❌ GROMACS $version: BUILD FAILED"
        fi
    done
    
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    
    # Final summary
    print_info ""
    print_info "================================================================"
    print_info "REBUILD ALL VERSIONS COMPLETED"
    print_info "================================================================"
    print_info "Total time: $(($total_time / 3600))h $(((total_time % 3600) / 60))m $((total_time % 60))s"
    print_info ""
    
    if [[ ${#successful_builds[@]} -gt 0 ]]; then
        print_success "Successful builds (${#successful_builds[@]}):"
        for version in "${successful_builds[@]}"; do
            print_success "  ✅ grinn:gromacs-$version"
        done
        print_info ""
    fi
    
    if [[ ${#failed_builds[@]} -gt 0 ]]; then
        print_error "Failed builds (${#failed_builds[@]}):"
        for version in "${failed_builds[@]}"; do
            print_error "  ❌ GROMACS $version"
        done
        print_info ""
        print_error "Some builds failed. Check the output above for details."
        exit 1
    else
        print_success "All ${#successful_builds[@]} versions built successfully!"
        
        print_info "Available images:"
        for version in "${successful_builds[@]}"; do
            print_info "  docker run --rm grinn:gromacs-$version gmx --version"
        done
    fi
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
    echo "  --no-cache        Build without using Docker cache"
    echo "  --rebuild-all     Rebuild all supported GROMACS versions"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 2024.1"
    echo "  $0 2025.2 --no-cache"
    echo "  $0 2020.7"
    echo "  $0 --rebuild-all             # Build all supported versions"
    echo "  $0 --rebuild-all --no-cache  # Build all without cache"
    echo ""
}

# Parse arguments
REBUILD_ALL=false

# Check for special flags first
if [[ "$1" == "--rebuild-all" ]]; then
    REBUILD_ALL=true
    shift
    
    # Parse remaining options for rebuild-all
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
                print_error "Unknown option for --rebuild-all: $1"
                print_info "Valid options: --no-cache, --help"
                exit 1
                ;;
        esac
    done
    
    # Execute rebuild all and exit (but don't call it here, set flag instead)
    # The actual function call will happen in main() after all functions are defined
fi

# Only check for missing arguments if we're not doing rebuild-all
if [[ "$REBUILD_ALL" != "true" ]]; then
    if [ $# -eq 0 ]; then
        print_error "No GROMACS version specified"
        show_usage
        exit 1
    fi
    
    # Check for help first
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        show_usage
        exit 0
    fi
    
    GROMACS_VERSION="$1"
    shift
    
    # Parse options
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
fi

# Find the grinn repository root directory
find_grinn_root() {
    local current_dir="$(pwd)"
    
    # Check if we're in the grinn directory or a parent directory containing it
    while [[ "$current_dir" != "/" ]]; do
        if [[ -f "$current_dir/grinn_workflow.py" && -f "$current_dir/Dockerfile" ]]; then
            echo "$current_dir"
            return 0
        fi
        
        if [[ -f "$current_dir/grinn/grinn_workflow.py" && -f "$current_dir/grinn/Dockerfile" ]]; then
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
        DOCKERFILE_PATH="$BUILD_CONTEXT/${GRINN_PREFIX}Dockerfile"
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

# Silent version of build_image for rebuild-all (returns success/failure)
build_image_silent() {
    # Build command
    BUILD_CMD="docker build"
    
    if [[ -n "$NO_CACHE" ]]; then
        BUILD_CMD="$BUILD_CMD $NO_CACHE"
    fi
    
    if [[ "$GROMACS_VERSION" != "2020.7" ]]; then
        BUILD_CMD="$BUILD_CMD --build-arg GROMACS_VERSION=$GROMACS_VERSION"
    fi
    
    BUILD_CMD="$BUILD_CMD --build-arg GRINN_PREFIX=$GRINN_PREFIX"
    BUILD_CMD="$BUILD_CMD -f $DOCKERFILE_PATH"
    BUILD_CMD="$BUILD_CMD -t $IMAGE_TAG"
    BUILD_CMD="$BUILD_CMD $BUILD_CONTEXT"
    
    # Execute build silently
    eval "$BUILD_CMD" >/dev/null 2>&1
    return $?
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

# Silent version of verify_image for rebuild-all (returns success/failure)
verify_image_silent() {
    if docker run --rm "$IMAGE_TAG" gmx --version 2>/dev/null | grep -q "$GROMACS_VERSION"; then
        if docker run --rm "$IMAGE_TAG" workflow --help >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
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
    # Check if rebuild-all flag is set
    if [[ "$REBUILD_ALL" == "true" ]]; then
        rebuild_all_versions "$NO_CACHE"
        exit 0
    fi
    
    # Normal single-version build
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
