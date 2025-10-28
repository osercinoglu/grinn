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

# Validate GROMACS version (exclude 2020.7 which needs special handling, allow NONE for dashboard-only)
RUN if [ "${GROMACS_VERSION}" = "NONE" ]; then \
        echo "📊 Building dashboard-only image (no GROMACS compilation)"; \
    else \
        GROMACS_MAJOR=$(echo "${GROMACS_VERSION}" | cut -d. -f1) && \
        GROMACS_MINOR=$(echo "${GROMACS_VERSION}" | cut -d. -f2) && \
        if [ "${GROMACS_MAJOR}" = "2020" ]; then \
            echo "❌ GROMACS 2020.7 requires Ubuntu 20.04 and special handling"; \
            echo "Please use the legacy Dockerfile: Dockerfile.gromacs-2020.7"; \
            exit 1; \
        elif [ "${GROMACS_MAJOR}" -lt "2021" ]; then \
            echo "❌ Unsupported GROMACS version: ${GROMACS_VERSION}"; \
            echo "Supported versions: 2021.7, 2022.x, 2023.x, 2024.x, 2025.x, or NONE for dashboard-only"; \
            exit 1; \
        fi && \
        echo "✅ Building GROMACS ${GROMACS_VERSION} on Ubuntu 22.04"; \
    fi

# Install build dependencies (only if building GROMACS)
RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        echo "📦 Installing GROMACS build dependencies..."; \
        apt-get update && apt-get install -y \
            build-essential \
            wget \
            libfftw3-dev \
            libopenmpi-dev \
            libhwloc-dev \
            libblas-dev \
            liblapack-dev \
            pkg-config \
            && rm -rf /var/lib/apt/lists/*; \
    else \
        echo "📦 Skipping GROMACS build dependencies (dashboard-only mode)"; \
        apt-get update && apt-get install -y \
            wget \
            curl \
            && rm -rf /var/lib/apt/lists/*; \
    fi

# Conditional CMake installation based on GROMACS version (skip for dashboard-only)
# GROMACS 2025+ requires modern CMake 3.28+
RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        GROMACS_MAJOR=$(echo "${GROMACS_VERSION}" | cut -d. -f1) && \
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
        fi; \
    else \
        echo "📦 Skipping CMake installation (dashboard-only mode)"; \
    fi

# Download and build GROMACS (skip for dashboard-only mode)
WORKDIR /tmp/gromacs-build

RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        echo "🔬 Building GROMACS version: ${GROMACS_VERSION}" && \
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
        echo "✅ GROMACS ${GROMACS_VERSION} built successfully"; \
    else \
        echo "📊 Skipping GROMACS build (dashboard-only mode)" && \
        cd / && rm -rf /tmp/gromacs-build && \
        mkdir -p /usr/local/gromacs/bin && \
        echo "#!/bin/bash" > /usr/local/gromacs/bin/gmx && \
        echo 'echo "❌ GROMACS not available in dashboard-only mode"' >> /usr/local/gromacs/bin/gmx && \
        echo 'exit 1' >> /usr/local/gromacs/bin/gmx && \
        chmod +x /usr/local/gromacs/bin/gmx; \
    fi

# ============================================================================
# STAGE 2: Python Environment (cached independently of gRINN code changes)
# ============================================================================

# Use Ubuntu 22.04 for runtime (matches builder stage)
FROM ubuntu:22.04 as python-env

# Re-declare ARG
ARG GROMACS_VERSION=2024.1

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ALWAYS_YES=true
ENV MAMBA_NO_BANNER=1

# Install runtime dependencies (conditional on GROMACS vs dashboard-only)
RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        echo "📦 Installing full runtime dependencies (with GROMACS support)..."; \
        apt-get update && apt-get install -y \
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
            && rm -rf /var/lib/apt/lists/*; \
    else \
        echo "📦 Installing minimal runtime dependencies (dashboard-only)..."; \
        apt-get update && apt-get install -y \
            python3 \
            python3-pip \
            python3-dev \
            wget \
            curl \
            && rm -rf /var/lib/apt/lists/*; \
    fi

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Miniforge (conda-forge based, no Anaconda ToS issues)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        MINIFORGE_ARCH="aarch64"; \
    else \
        MINIFORGE_ARCH="x86_64"; \
    fi && \
    wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# Configure conda to avoid ToS issues and use only open source channels
RUN conda config --set solver libmamba && \
    conda config --set remote_read_timeout_secs 120 && \
    conda config --set always_yes true && \
    conda config --show-sources

# Set conda to use conda-forge as primary channel and avoid anaconda ToS issues
RUN conda config --remove channels defaults 2>/dev/null || true && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    echo "channels:" > ~/.condarc && \
    echo "  - conda-forge" >> ~/.condarc && \
    echo "  - bioconda" >> ~/.condarc && \
    echo "  - plotly" >> ~/.condarc && \
    echo "channel_priority: strict" >> ~/.condarc

# Install mamba for faster package installation (from conda-forge only)
RUN echo "📦 Installing mamba..." && \
    CONDA_ALWAYS_YES=yes conda install -c conda-forge mamba && \
    echo "✅ Mamba installed successfully" && \
    mamba --version

# Create conda environment (conditional dependencies based on mode)
RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        echo "📦 Creating full gRINN environment with GROMACS support..."; \
        echo "Debug: Creating base environment..." && \
        CONDA_ALWAYS_YES=yes mamba create -n grinn-env -c conda-forge python=3.10 && \
        echo "Debug: Installing core scientific packages..." && \
        CONDA_ALWAYS_YES=yes mamba install -n grinn-env -c conda-forge \
            numpy \
            scipy \
            pandas \
            networkx \
            tqdm && \
        echo "Debug: Installing bio packages..." && \
        CONDA_ALWAYS_YES=yes mamba install -n grinn-env -c conda-forge -c bioconda \
            pdbfixer \
            prody \
            mdtraj \
            openmm \
            biotite && \
        echo "Debug: Installing gromacs wrapper..." && \
        CONDA_ALWAYS_YES=yes mamba install -n grinn-env -c conda-forge \
            gromacswrapper \
            panedr && \
        echo "Debug: Installing web packages..." && \
        CONDA_ALWAYS_YES=yes mamba install -n grinn-env -c conda-forge -c plotly \
            dash \
            dash-bootstrap-components \
            plotly && \
        echo "Debug: Installing remaining packages..." && \
        CONDA_ALWAYS_YES=yes mamba install -n grinn-env -c conda-forge \
            pyprind; \
    else \
        echo "📦 Creating dashboard-only environment..."; \
        CONDA_ALWAYS_YES=yes mamba create -n grinn-env -c conda-forge python=3.10 && \
        CONDA_ALWAYS_YES=yes mamba install -n grinn-env -c conda-forge \
            numpy \
            scipy \
            pandas \
            networkx \
            tqdm \
            pyprind \
            gromacswrapper \
            panedr && \
        CONDA_ALWAYS_YES=yes mamba install -n grinn-env -c conda-forge -c bioconda \
            prody \
            mdtraj \
            pdbfixer \
            openmm && \
        CONDA_ALWAYS_YES=yes mamba install -n grinn-env -c conda-forge -c plotly \
            dash \
            dash-bootstrap-components \
            plotly; \
    fi

# Install additional packages via pip (more reliable for some packages)
RUN echo "📦 Installing additional packages via pip..." && \
    conda run -n grinn-env pip install --no-cache-dir \
        dash-molstar && \
    echo "✅ Pip packages installed successfully"

# Copy GROMACS installation from builder stage (create empty dir for dashboard-only)
RUN if [ "${GROMACS_VERSION}" = "NONE" ]; then \
        echo "📦 Creating placeholder GROMACS directory (dashboard-only mode)"; \
        mkdir -p /usr/local/gromacs/bin; \
    fi

# Copy GROMACS installation (will copy empty dir for dashboard-only)
COPY --from=gromacs-builder /usr/local/gromacs /usr/local/gromacs

# Set GROMACS environment (conditional)
RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        echo 'export PATH="/usr/local/gromacs/bin:$PATH"' >> /etc/environment && \
        echo 'export LD_LIBRARY_PATH="/usr/local/gromacs/lib:$LD_LIBRARY_PATH"' >> /etc/environment; \
    fi

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

# Set up GromacsWrapper config (only if GROMACS is available)
RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        echo "📦 Setting up GromacsWrapper..."; \
        conda run -n grinn-env python -c "import gromacs; gromacs.config.setup()"; \
    else \
        echo "📦 Skipping GromacsWrapper setup (dashboard-only mode)"; \
    fi

# Copy MDP files (only if building full gRINN)
RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        echo "📦 MDP files will be copied for full gRINN build"; \
    else \
        mkdir -p ./mdp_files; \
        echo "📦 Created empty MDP directory (dashboard-only mode)"; \
    fi

COPY ${GRINN_PREFIX}mdp_files ./mdp_files

# Copy source code conditionally
RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        echo "📦 Full gRINN workflow will be available"; \
    else \
        echo "📦 Dashboard-only mode - workflow disabled"; \
    fi

COPY ${GRINN_PREFIX}grinn_workflow.py ./
COPY ${GRINN_PREFIX}gRINN_Dashboard ./gRINN_Dashboard

# Make grinn_workflow.py executable and add dashboard-only guard
RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        chmod +x grinn_workflow.py; \
    else \
        # For dashboard-only mode, add a runtime guard at the top of main execution
        # but keep the file importable for dashboard
        sed -i '1s/^/import sys\n/' grinn_workflow.py && \
        sed -i '/if __name__ == .__main__./a\    print("❌ gRINN workflow not available in dashboard-only mode")\n    print("This image was built for dashboard functionality only")\n    print("Use: docker run -p 8051:8051 <image> dashboard <results_folder>")\n    sys.exit(1)' grinn_workflow.py && \
        chmod +x grinn_workflow.py; \
    fi

# Create sophisticated entrypoint script
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Function to show usage' >> /app/entrypoint.sh && \
    echo 'show_usage() {' >> /app/entrypoint.sh && \
    if [ "${GROMACS_VERSION}" = "NONE" ]; then \
        echo '    echo "gRINN Dashboard Container (Dashboard-Only Mode)"' >> /app/entrypoint.sh; \
        echo '    echo ""' >> /app/entrypoint.sh; \
        echo '    echo "  docker run [docker-options] grinn-dashboard:latest <mode> [mode-options]"' >> /app/entrypoint.sh; \
        echo '    echo ""' >> /app/entrypoint.sh; \
        echo '    echo "Available modes:"' >> /app/entrypoint.sh; \
        echo '    echo "  dashboard <results_folder>                  - Launch interactive dashboard"' >> /app/entrypoint.sh; \
        echo '    echo "  bash                                        - Start interactive bash session"' >> /app/entrypoint.sh; \
        echo '    echo "  help                                        - Show this help message"' >> /app/entrypoint.sh; \
        echo '    echo ""' >> /app/entrypoint.sh; \
        echo '    echo "Examples:"' >> /app/entrypoint.sh; \
        echo '    echo "  docker run -p 8051:8051 -v /data:/data grinn-dashboard:latest dashboard /data/results"' >> /app/entrypoint.sh; \
        echo '    echo "  docker run -it grinn-dashboard:latest bash"' >> /app/entrypoint.sh; \
        echo '    echo ""' >> /app/entrypoint.sh; \
        echo '    echo "Note: This is a dashboard-only image. For full gRINN workflow, use grinn:gromacs-VERSION"' >> /app/entrypoint.sh; \
    else \
        echo '    echo "gRINN Docker Container - GROMACS ${GROMACS_VERSION}"' >> /app/entrypoint.sh; \
        echo '    echo ""' >> /app/entrypoint.sh; \
        echo '    echo "  docker run [docker-options] grinn:gromacs-${GROMACS_VERSION} <mode> [mode-options]"' >> /app/entrypoint.sh; \
        echo '    echo ""' >> /app/entrypoint.sh; \
        echo '    echo "Available modes:"' >> /app/entrypoint.sh; \
        echo '    echo "  workflow <input.pdb> <output_dir> [options]  - Run gRINN workflow analysis"' >> /app/entrypoint.sh; \
        echo '    echo "  dashboard <results_folder>                  - Launch interactive dashboard"' >> /app/entrypoint.sh; \
        echo '    echo "  gmx <gmx_command> [gmx_options]             - Run GROMACS commands"' >> /app/entrypoint.sh; \
        echo '    echo "  bash                                        - Start interactive bash session"' >> /app/entrypoint.sh; \
        echo '    echo "  help                                        - Show this help message"' >> /app/entrypoint.sh; \
        echo '    echo ""' >> /app/entrypoint.sh; \
        echo '    echo "Examples:"' >> /app/entrypoint.sh; \
        echo '    echo "  docker run -v /data:/data grinn:gromacs-${GROMACS_VERSION} workflow /data/protein.pdb /data/results"' >> /app/entrypoint.sh; \
        echo '    echo "  docker run -p 8051:8051 -v /data:/data grinn:gromacs-${GROMACS_VERSION} dashboard /data/results"' >> /app/entrypoint.sh; \
        echo '    echo "  docker run -v /data:/data grinn:gromacs-${GROMACS_VERSION} gmx --version"' >> /app/entrypoint.sh; \
        echo '    echo "  docker run -it grinn:gromacs-${GROMACS_VERSION} bash"' >> /app/entrypoint.sh; \
    fi && \
    echo '}' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Source GROMACS environment (only if available)' >> /app/entrypoint.sh && \
    echo 'if [ -f "$GMXRC_PATH" ]; then' >> /app/entrypoint.sh && \
    echo '    source $GMXRC_PATH' >> /app/entrypoint.sh && \
    echo 'fi' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Handle different execution modes' >> /app/entrypoint.sh && \
    echo 'case "$1" in' >> /app/entrypoint.sh && \
    echo '    "workflow")' >> /app/entrypoint.sh && \
    if [ "${GROMACS_VERSION}" = "NONE" ]; then \
        echo '        echo "❌ gRINN workflow not available in dashboard-only mode"' >> /app/entrypoint.sh; \
        echo '        echo "This image was built for dashboard functionality only."' >> /app/entrypoint.sh; \
        echo '        echo "For full gRINN workflow, use: grinn:gromacs-VERSION"' >> /app/entrypoint.sh; \
        echo '        exit 1' >> /app/entrypoint.sh; \
    else \
        echo '        shift' >> /app/entrypoint.sh; \
        echo '        echo "🧬 Starting gRINN Workflow Analysis with GROMACS ${GROMACS_VERSION}..."' >> /app/entrypoint.sh; \
        echo '        echo "Arguments: $@"' >> /app/entrypoint.sh; \
        echo '        # Enable real-time output with multiple techniques' >> /app/entrypoint.sh; \
        echo '        export PYTHONUNBUFFERED=1' >> /app/entrypoint.sh; \
        echo '        export PYTHONIOENCODING=utf-8' >> /app/entrypoint.sh; \
        echo '        stdbuf -oL -eL conda run -n grinn-env python -u grinn_workflow.py "$@"' >> /app/entrypoint.sh; \
    fi && \
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
    if [ "${GROMACS_VERSION}" = "NONE" ]; then \
        echo '        echo "❌ GROMACS not available in dashboard-only mode"' >> /app/entrypoint.sh; \
        echo '        echo "This image was built for dashboard functionality only."' >> /app/entrypoint.sh; \
        echo '        echo "For GROMACS commands, use: grinn:gromacs-VERSION"' >> /app/entrypoint.sh; \
        echo '        exit 1' >> /app/entrypoint.sh; \
    else \
        echo '        shift' >> /app/entrypoint.sh; \
        echo '        echo "⚗️  Running GROMACS ${GROMACS_VERSION} command: gmx $@"' >> /app/entrypoint.sh; \
        echo '        # Enable real-time output for GROMACS commands' >> /app/entrypoint.sh; \
        echo '        stdbuf -oL -eL gmx "$@"' >> /app/entrypoint.sh; \
    fi && \
    echo '        ;;' >> /app/entrypoint.sh && \
    echo '    "bash")' >> /app/entrypoint.sh && \
    echo '        echo "🐚 Starting interactive bash session..."' >> /app/entrypoint.sh && \
    if [ "${GROMACS_VERSION}" = "NONE" ]; then \
        echo '        echo "Dashboard-only mode - no GROMACS available"' >> /app/entrypoint.sh; \
    else \
        echo '        echo "GROMACS ${GROMACS_VERSION} environment is already sourced"' >> /app/entrypoint.sh; \
    fi && \
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

# Verify GROMACS installation (only if available)
RUN if [ "${GROMACS_VERSION}" != "NONE" ]; then \
        echo "🔬 Verifying GROMACS installation..."; \
        gmx --version; \
    else \
        echo "📊 Dashboard-only mode - no GROMACS to verify"; \
    fi

# Expose port for dashboard
EXPOSE 8051

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
