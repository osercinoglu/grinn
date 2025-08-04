# Unified gRINN Dockerfile with Multi-Stage Build for Optimal Caching
# Supports GROMACS versions 2021.7 through 2025.2 with conditional logic
# 
# For GROMACS 2020.7, use the legacy Dockerfile (requires Ubuntu 20.04)
#
# Usage:
#   docker build --build-arg GROMACS_VERSION=2024.1 -t grinn:gromacs-2024.1 .
#   docker build --build-arg GROMACS_VERSION=2025.2 -t grinn:gromacs-2025.2 .
#   docker build --build-arg GROMACS_VERSION=2022.4 -t grinn:gromacs-2022.4 .

# ============================================================================
# STAGE 1: GROMACS Builder (cached independently of gRINN code changes)
# ============================================================================

# Build arguments
ARG GROMACS_VERSION=2024.1

# Modern GROMACS versions use Ubuntu 22.04
FROM ubuntu:22.04 as gromacs-builder

# Re-declare ARG after FROM
ARG GROMACS_VERSION=2024.1

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Validate GROMACS version (exclude 2020.7 which needs special handling)
RUN GROMACS_MAJOR=$(echo "${GROMACS_VERSION}" | cut -d. -f1) && \
    GROMACS_MINOR=$(echo "${GROMACS_VERSION}" | cut -d. -f2) && \
    if [ "${GROMACS_MAJOR}" = "2020" ]; then \
        echo "❌ GROMACS 2020.7 requires Ubuntu 20.04 and special handling"; \
        echo "Please use the legacy Dockerfile: Dockerfile.gromacs-2020.7"; \
        exit 1; \
    elif [ "${GROMACS_MAJOR}" -lt "2021" ]; then \
        echo "❌ Unsupported GROMACS version: ${GROMACS_VERSION}"; \
        echo "Supported versions: 2021.7, 2022.x, 2023.x, 2024.x, 2025.x"; \
        exit 1; \
    fi && \
    echo "✅ Building GROMACS ${GROMACS_VERSION} on Ubuntu 22.04"

# Install build dependencies (common for all versions)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libfftw3-dev \
    libopenmpi-dev \
    libhwloc-dev \
    libblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Conditional CMake installation based on GROMACS version
# GROMACS 2025+ requires modern CMake 3.28+
RUN GROMACS_MAJOR=$(echo "${GROMACS_VERSION}" | cut -d. -f1) && \
    if [ "${GROMACS_MAJOR}" -ge "2025" ]; then \
        echo "📦 Installing modern CMake 3.28+ for GROMACS ${GROMACS_VERSION}..." && \
        apt-get update && \
        apt-get install -y gnupg && \
        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
        echo 'deb https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
        apt-get update && \
        apt-get install -y cmake && \
        rm -rf /var/lib/apt/lists/* && \
        cmake --version; \
    else \
        echo "📦 Installing system CMake for GROMACS ${GROMACS_VERSION}..." && \
        apt-get update && \
        apt-get install -y cmake && \
        rm -rf /var/lib/apt/lists/* && \
        cmake --version; \
    fi

# Download and build GROMACS
WORKDIR /tmp/gromacs-build

RUN echo "🔬 Building GROMACS version: ${GROMACS_VERSION}" && \
    wget ftp://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz && \
    tar xfz gromacs-${GROMACS_VERSION}.tar.gz && \
    cd gromacs-${GROMACS_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. \
        -DGMX_BUILD_OWN_FFTW=ON \
        -DGMX_GPU=OFF \
        -DGMX_MPI=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
        -DCMAKE_BUILD_TYPE=Release \
        -DGMX_USE_OPENCL=OFF && \
    make -j$(nproc) && \
    make install && \
    # Cleanup build artifacts to reduce layer size
    cd / && rm -rf /tmp/gromacs-build && \
    echo "✅ GROMACS ${GROMACS_VERSION} built successfully"

# ============================================================================
# STAGE 2: Python Environment (cached independently of gRINN code changes)
# ============================================================================

# Use Ubuntu 22.04 for runtime (matches builder stage)
FROM ubuntu:22.04 as python-env

# Re-declare ARG
ARG GROMACS_VERSION=2024.1

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libfftw3-3 \
    libopenmpi3 \
    libhwloc15 \
    libblas3 \
    liblapack3 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Miniconda
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        MINICONDA_ARCH="aarch64"; \
    else \
        MINICONDA_ARCH="x86_64"; \
    fi && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Configure conda channels
RUN conda config --set channel_priority strict && \
    conda config --add channels conda-forge && \
    conda config --add channels bioconda && \
    conda config --add channels plotly

# Accept conda Terms of Service for required channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda tos accept --override-channels --channel conda-forge

# Install mamba for faster package installation
RUN conda install -y -c conda-forge mamba

# Create conda environment with all gRINN dependencies
RUN mamba create -y -n grinn-env -c conda-forge -c bioconda -c plotly \
    python=3.10 \
    pdbfixer \
    prody \
    numpy \
    scipy \
    pyprind \
    pandas \
    mdtraj \
    openmm \
    panedr \
    gromacswrapper \
    networkx \
    tqdm \
    dash \
    dash-bootstrap-components \
    plotly

# Install dash-molstar for 3D visualization
RUN conda run -n grinn-env pip install dash-molstar

# Copy GROMACS installation from builder stage
COPY --from=gromacs-builder /usr/local/gromacs /usr/local/gromacs

# Set GROMACS environment
ENV PATH="/usr/local/gromacs/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/gromacs/lib:$LD_LIBRARY_PATH"

# ============================================================================
# STAGE 3: Final gRINN Application (rebuilds when code changes)
# ============================================================================

FROM python-env as final

# Re-declare build args
ARG GROMACS_VERSION=2024.1
ARG GRINN_PREFIX=""

# Add labels for better image management
LABEL maintainer="gRINN Team"
LABEL gromacs.version="${GROMACS_VERSION}"
LABEL description="gRINN molecular dynamics analysis with GROMACS ${GROMACS_VERSION}"

# Set working directory
WORKDIR /app

# Set Python environment variables for real-time output in Docker containers
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=utf-8

# Set GROMACS environment variables
ENV GMX_MAXBACKUP=-1
ENV GMXRC_PATH=/usr/local/gromacs/bin/GMXRC

# Activate conda env and source GMXRC for all future RUN/CMD
SHELL ["/bin/bash", "-c"]

# Set up GromacsWrapper config
RUN conda run -n grinn-env python -c "import gromacs; gromacs.config.setup()"

# Copy MDP files (these change less frequently)
COPY ${GRINN_PREFIX}mdp_files ./mdp_files

# Copy gRINN source code and dashboard (these change more frequently)
COPY ${GRINN_PREFIX}grinn_workflow.py ./
COPY ${GRINN_PREFIX}gRINN_Dashboard ./gRINN_Dashboard

# Make grinn_workflow.py executable
RUN chmod +x grinn_workflow.py

# Create sophisticated entrypoint script
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Function to show usage' >> /app/entrypoint.sh && \
    echo 'show_usage() {' >> /app/entrypoint.sh && \
    echo '    echo "gRINN Docker Container - GROMACS ${GROMACS_VERSION}"' >> /app/entrypoint.sh && \
    echo '    echo ""' >> /app/entrypoint.sh && \
    echo '    echo "  docker run [docker-options] grinn:gromacs-${GROMACS_VERSION} <mode> [mode-options]"' >> /app/entrypoint.sh && \
    echo '    echo ""' >> /app/entrypoint.sh && \
    echo '    echo "Available modes:"' >> /app/entrypoint.sh && \
    echo '    echo "  workflow <input.pdb> <output_dir> [options]  - Run gRINN workflow analysis"' >> /app/entrypoint.sh && \
    echo '    echo "  dashboard <results_folder>                  - Launch interactive dashboard"' >> /app/entrypoint.sh && \
    echo '    echo "  gmx <gmx_command> [gmx_options]             - Run GROMACS commands"' >> /app/entrypoint.sh && \
    echo '    echo "  bash                                        - Start interactive bash session"' >> /app/entrypoint.sh && \
    echo '    echo "  help                                        - Show this help message"' >> /app/entrypoint.sh && \
    echo '    echo ""' >> /app/entrypoint.sh && \
    echo '    echo "Examples:"' >> /app/entrypoint.sh && \
    echo '    echo "  docker run -v /data:/data grinn:gromacs-${GROMACS_VERSION} workflow /data/protein.pdb /data/results"' >> /app/entrypoint.sh && \
    echo '    echo "  docker run -p 8051:8051 -v /data:/data grinn:gromacs-${GROMACS_VERSION} dashboard /data/results"' >> /app/entrypoint.sh && \
    echo '    echo "  docker run -v /data:/data grinn:gromacs-${GROMACS_VERSION} gmx --version"' >> /app/entrypoint.sh && \
    echo '    echo "  docker run -it grinn:gromacs-${GROMACS_VERSION} bash"' >> /app/entrypoint.sh && \
    echo '}' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Source GROMACS environment' >> /app/entrypoint.sh && \
    echo 'source $GMXRC_PATH' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Handle different execution modes' >> /app/entrypoint.sh && \
    echo 'case "$1" in' >> /app/entrypoint.sh && \
    echo '    "workflow")' >> /app/entrypoint.sh && \
    echo '        shift' >> /app/entrypoint.sh && \
    echo '        echo "🧬 Starting gRINN Workflow Analysis with GROMACS ${GROMACS_VERSION}..."' >> /app/entrypoint.sh && \
    echo '        echo "Arguments: $@"' >> /app/entrypoint.sh && \
    echo '        # Enable real-time output with multiple techniques' >> /app/entrypoint.sh && \
    echo '        export PYTHONUNBUFFERED=1' >> /app/entrypoint.sh && \
    echo '        export PYTHONIOENCODING=utf-8' >> /app/entrypoint.sh && \
    echo '        stdbuf -oL -eL conda run -n grinn-env python -u grinn_workflow.py "$@"' >> /app/entrypoint.sh && \
    echo '        ;;' >> /app/entrypoint.sh && \
    echo '    "dashboard")' >> /app/entrypoint.sh && \
    echo '        shift' >> /app/entrypoint.sh && \
    echo '        echo "📊 Starting gRINN Dashboard with GROMACS ${GROMACS_VERSION}..."' >> /app/entrypoint.sh && \
    echo '        echo "Dashboard will be available at http://localhost:8051"' >> /app/entrypoint.sh && \
    echo '        echo "Results folder: $1"' >> /app/entrypoint.sh && \
    echo '        # Enable real-time output with multiple techniques' >> /app/entrypoint.sh && \
    echo '        export PYTHONUNBUFFERED=1' >> /app/entrypoint.sh && \
    echo '        export PYTHONIOENCODING=utf-8' >> /app/entrypoint.sh && \
    echo '        stdbuf -oL -eL conda run -n grinn-env python -u gRINN_Dashboard/grinn_dashboard.py "$@"' >> /app/entrypoint.sh && \
    echo '        ;;' >> /app/entrypoint.sh && \
    echo '    "gmx")' >> /app/entrypoint.sh && \
    echo '        shift' >> /app/entrypoint.sh && \
    echo '        echo "⚗️  Running GROMACS ${GROMACS_VERSION} command: gmx $@"' >> /app/entrypoint.sh && \
    echo '        # Enable real-time output for GROMACS commands' >> /app/entrypoint.sh && \
    echo '        stdbuf -oL -eL gmx "$@"' >> /app/entrypoint.sh && \
    echo '        ;;' >> /app/entrypoint.sh && \
    echo '    "bash")' >> /app/entrypoint.sh && \
    echo '        echo "🐚 Starting interactive bash session..."' >> /app/entrypoint.sh && \
    echo '        echo "GROMACS ${GROMACS_VERSION} environment is already sourced"' >> /app/entrypoint.sh && \
    echo '        echo "Conda environment: grinn-env"' >> /app/entrypoint.sh && \
    echo '        echo "To activate conda env: conda activate grinn-env"' >> /app/entrypoint.sh && \
    echo '        exec /bin/bash' >> /app/entrypoint.sh && \
    echo '        ;;' >> /app/entrypoint.sh && \
    echo '    "help"|"--help"|"-h")' >> /app/entrypoint.sh && \
    echo '        show_usage' >> /app/entrypoint.sh && \
    echo '        ;;' >> /app/entrypoint.sh && \
    echo '    "")' >> /app/entrypoint.sh && \
    echo '        echo "❌ Error: No execution mode specified"' >> /app/entrypoint.sh && \
    echo '        echo ""' >> /app/entrypoint.sh && \
    echo '        show_usage' >> /app/entrypoint.sh && \
    echo '        exit 1' >> /app/entrypoint.sh && \
    echo '        ;;' >> /app/entrypoint.sh && \
    echo '    *)' >> /app/entrypoint.sh && \
    echo '        echo "❌ Error: Unknown execution mode: $1"' >> /app/entrypoint.sh && \
    echo '        echo ""' >> /app/entrypoint.sh && \
    echo '        show_usage' >> /app/entrypoint.sh && \
    echo '        exit 1' >> /app/entrypoint.sh && \
    echo '        ;;' >> /app/entrypoint.sh && \
    echo 'esac' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Verify GROMACS installation
RUN gmx --version

# Expose port for dashboard
EXPOSE 8051

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
