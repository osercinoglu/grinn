# gRINN Multi-Version Docker Support

This directory contains Docker configurations that support building gRINN with different GROMACS versions, providing flexibility for users with diverse computational requirements.

## Quick Start

### Build with Default GROMACS Version (2024.1)
```bash
./build-docker.sh
```

### Build with Specific GROMACS Version
```bash
./build-docker.sh --version 2023.3
./build-docker.sh --version 2024.1
./build-docker.sh --version 2024.2
```

### Build for Apple Silicon (M1/M2 Macs)
```bash
./build-docker.sh --platform linux/arm64 --version 2024.1
```

### Build Development Version
```bash
./build-docker.sh --dev --version 2023.3
```

## Supported GROMACS Versions

| Version | Status | Notes |
|---------|--------|-------|
| 2020.6  | ✅ Supported | Stable, older version |
| 2021.6  | ✅ Supported | Stable release |
| 2022.5  | ✅ Supported | Stable release |
| 2023.3  | ✅ Supported | Stable release |
| 2024.1  | ✅ Supported | **Default** - Latest stable |
| 2024.2  | ✅ Supported | Latest development |

## Available Docker Images

### Production Images (Dockerfile)
- Optimized for production use
- No development tools
- Smaller image size
- Pre-built test data included

### Development Images (Dockerfile.dev)
- Includes development tools
- Faster iteration during development
- Skips some validation steps
- Larger image size

## Usage Examples

### 1. Basic Workflow Execution
```bash
# Using default GROMACS 2024.1
docker run -v /path/to/data:/data grinn:gromacs-2024.1 \
    workflow /data/protein.pdb /data/results

# Using specific GROMACS version
docker run -v /path/to/data:/data grinn:gromacs-2023.3 \
    workflow /data/protein.pdb /data/results --gpu
```

### 2. Interactive Dashboard
```bash
docker run -p 8051:8051 -v /path/to/results:/data \
    grinn:gromacs-2024.1 dashboard /data
```

### 3. Interactive Development
```bash
docker run -it -v /path/to/data:/data \
    grinn:gromacs-2024.1-dev bash
```

### 4. Custom GROMACS Commands
```bash
docker run -v /path/to/data:/data grinn:gromacs-2024.1 \
    gmx pdb2gmx -f /data/protein.pdb -p /data/topol.top
```

## Build Script Options

```bash
./build-docker.sh [OPTIONS]

Options:
  -v, --version GROMACS_VERSION    GROMACS version to build (default: 2024.1)
  -t, --tag IMAGE_TAG              Docker image tag (default: grinn)
  -d, --dev                        Build development version (Dockerfile.dev)
  -p, --platform PLATFORM         Build for specific platform
  -l, --list                       List available GROMACS versions
  -h, --help                       Show help message
```

## Manual Docker Build

If you prefer to build manually without the script:

### Build with Custom GROMACS Version
```bash
# Production image with GROMACS 2023.3
docker build --build-arg GROMACS_VERSION=2023.3 \
    -t grinn:gromacs-2023.3 .

# Development image with GROMACS 2024.1
docker build --build-arg GROMACS_VERSION=2024.1 \
    -f Dockerfile.dev -t grinn:gromacs-2024.1-dev .

# For Apple Silicon (ARM64)
docker build --platform linux/arm64 \
    --build-arg GROMACS_VERSION=2024.1 \
    -t grinn:gromacs-2024.1-arm64 .
```

## Architecture Support

### x86_64 (Intel/AMD)
- ✅ Fully supported
- All GROMACS versions work
- Best performance for CPU-intensive tasks

### ARM64 (Apple Silicon M1/M2)
- ✅ Supported with platform flag
- Requires `--platform linux/arm64`
- May have slightly different performance characteristics

## Version Compatibility

### GROMACS Version Selection Guide

**For Maximum Compatibility:**
- Use GROMACS 2024.1 (default)
- Works with most force fields and file formats

**For Legacy Systems:**
- Use GROMACS 2020.6 or 2021.6
- Better compatibility with older topology files

**For Latest Features:**
- Use GROMACS 2024.2
- Latest optimizations and features

**For Specific Research Needs:**
- Match the GROMACS version used in your original simulations
- Ensures reproducibility

## Troubleshooting

### Build Issues

**Docker Build Fails:**
```bash
# Check Docker daemon
docker --version
docker info

# Clean Docker cache
docker system prune -f

# Try with no cache
docker build --no-cache --build-arg GROMACS_VERSION=2024.1 -t grinn:test .
```

**Platform Issues on Apple Silicon:**
```bash
# Force ARM64 build
./build-docker.sh --platform linux/arm64 --version 2024.1

# Check available platforms
docker buildx ls
```

**GROMACS Build Errors:**
```bash
# Try older version
./build-docker.sh --version 2023.3

# Check build logs
docker build --build-arg GROMACS_VERSION=2024.1 --progress=plain -t grinn:debug .
```

### Runtime Issues

**Memory Issues:**
- Increase Docker memory allocation
- Use smaller trajectory files for testing
- Enable frame skipping: `--skip 10`

**Permission Issues:**
```bash
# Fix file permissions
sudo chown -R $USER:$USER /path/to/data

# Run with user mapping
docker run --user $(id -u):$(id -g) -v /data:/data grinn:latest workflow ...
```

## Performance Considerations

### Image Size Comparison
- Base image: ~2-3 GB
- With GROMACS: ~3-4 GB per version
- Development images: ~4-5 GB per version

### Build Time Estimates
- GROMACS compilation: 20-60 minutes (depends on CPU)
- Total build time: 30-90 minutes
- ARM64 builds may take longer

### Runtime Performance
- Native platform builds perform best
- GPU acceleration requires compatible host drivers
- Memory usage scales with system size

## Best Practices

### For Production Use
1. Use specific version tags: `grinn:gromacs-2024.1`
2. Avoid `latest` tag for reproducibility
3. Pin to specific GROMACS versions in scripts
4. Use production Dockerfile for deployment

### For Development
1. Use development images for faster iteration
2. Mount source code for live editing
3. Use volume mounts for persistent data
4. Clean up unused images regularly

### For CI/CD
1. Build images in parallel for different versions
2. Tag images with build metadata
3. Use multi-stage builds to optimize size
4. Cache base layers across builds

## Contributing

When adding support for new GROMACS versions:

1. Update `build-docker.sh` supported versions list
2. Test build with new version
3. Update this documentation
4. Add version to CI/CD pipeline
5. Test compatibility with existing workflows

## License

This Docker configuration is part of the gRINN project and follows the same licensing terms.
