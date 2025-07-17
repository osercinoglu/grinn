FROM ubuntu:22.04

# System dependencies
RUN apt-get update && \
    apt-get install -y wget git build-essential cmake sudo python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda and Mamba
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Accept conda Terms of Service for default channels
RUN conda config --set channel_priority strict && \
    conda config --add channels conda-forge && \
    conda config --add channels bioconda && \
    conda config --add channels plotly && \
    echo "yes" | conda tos accept

RUN conda install -y -c conda-forge mamba

# Create environment with all dependencies including dashboard
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

# Install gsutil from Google Cloud SDK (system Python)
RUN apt-get update && \
    apt-get install -y curl apt-transport-https ca-certificates gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk

# Set up working directory
WORKDIR /app

# Copy your code, dashboard, and mdp_files
COPY grinn_workflow.py ./
COPY gRINN_Dashboard ./gRINN_Dashboard
COPY mdp_files ./mdp_files

# Download test data from public Google Cloud Storage bucket
RUN gsutil -m cp -r gs://grinn-test-data/* $PWD

# Set environment variables for GROMACS
ENV PATH="/usr/local/gromacs/bin:$PATH"
ENV GMX_MAXBACKUP=-1

# Activate conda env and source GMXRC for all future RUN/CMD
SHELL ["/bin/bash", "-c"]

ENV GMXRC_PATH=/usr/local/gromacs/bin/GMXRC

# Set up GromacsWrapper config in root's home
RUN conda run -n grinn-env python -c "import gromacs; gromacs.config.setup()"

# Copy test script into the image
COPY test.sh ./

# Make sure test.sh is executable
RUN chmod +x test.sh

# Run the test script during build with the correct conda environment
RUN set -x && conda run -n grinn-env bash ./test.sh

# Create a script to determine execution mode
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'if [ "$1" = "dashboard" ]; then' >> /app/entrypoint.sh && \
    echo '    shift' >> /app/entrypoint.sh && \
    echo '    source $GMXRC_PATH && conda run -n grinn-env python gRINN_Dashboard/grinn_dashboard.py "$@"' >> /app/entrypoint.sh && \
    echo 'else' >> /app/entrypoint.sh && \
    echo '    source $GMXRC_PATH && conda run -n grinn-env python grinn_workflow.py "$@"' >> /app/entrypoint.sh && \
    echo 'fi' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Expose port for dashboard
EXPOSE 8051

ENTRYPOINT ["/app/entrypoint.sh"]