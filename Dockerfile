FROM ubuntu:22.04

# System dependencies
RUN apt-get update && \
    apt-get install -y wget git build-essential cmake sudo python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda and Mamba
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda install -y -c conda-forge mamba

# Create environment with all dependencies
RUN mamba create -y -n grinn-env -c conda-forge -c bioconda \
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
    gromacswrapper

# Install GROMACS from source
WORKDIR /tmp
RUN wget ftp://ftp.gromacs.org/gromacs/gromacs-2024.1.tar.gz && \
    tar xfz gromacs-2024.1.tar.gz && \
    cd gromacs-2024.1 && \
    mkdir build && \
    cd build && \
    cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON && \
    make -j$(nproc) && \
    make check && \
    sudo make install && \
    cd / && \
    rm -rf /tmp/gromacs-2024.1*

# Set up working directory
WORKDIR /app

# Copy your code and mdp_files
COPY grinn_workflow.py ./
COPY mdp_files ./mdp_files

# Set environment variables for GROMACS
ENV PATH="/usr/local/gromacs/bin:$PATH"
ENV GMX_MAXBACKUP=-1

# Activate conda env for all future RUN/CMD
SHELL ["conda", "run", "-n", "grinn-env", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "-n", "grinn-env", "python", "grinn_workflow.py"]