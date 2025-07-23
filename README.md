# gRINN: Protein Energy Networks from Molecular Dynamics

[![License](https://img.shields.io/badge/License-Custom-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-brightgreen.svg)](https://python.org)
[![GROMACS](https://img.shields.io/badge/GROMACS-2024.1-orange.svg)](https://www.gromacs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

**gRINN** (get Residue Interaction Energies and Networks) is a computational tool for **residue interaction energy-based analysis** of protein molecular dynamics (MD) simulation trajectories. This is the **next-generation version** of gRINN, featuring significant performance improvements, enhanced workflow capabilities, and containerized deployment.

## 🚀 Quick Start

### Using Docker (Easiest)

```bash
# Build the image
docker build -t grinn .

# Get a protein structure
wget https://files.rcsb.org/download/1L2Y.pdb

# 1. Prepare protein structure with AMBER force field
docker run -v $(pwd):/data grinn gmx pdb2gmx -f /data/1L2Y.pdb -o /data/processed.gro -p /data/topol.top -ff amber99sb-ildn -water tip3p

# 2. Define simulation box
docker run -v $(pwd):/data grinn gmx editconf -f /data/processed.gro -o /data/boxed.gro -c -d 1.0 -bt cubic

# 3. Add solvent (optional but recommended)
docker run -v $(pwd):/data grinn gmx solvate -cp /data/boxed.gro -cs spc216.gro -o /data/solvated.gro -p /data/topol.top

# 4. Energy minimization
docker run -v $(pwd):/data grinn gmx grompp -f /app/mdp_files/minim.mdp -c /data/solvated.gro -p /data/topol.top -o /data/em.tpr
docker run -v $(pwd):/data grinn gmx mdrun -v -deffnm /data/em

# 5. Short MD simulation
docker run -v $(pwd):/data grinn gmx grompp -f /app/mdp_files/npt.mdp -c /data/em.gro -p /data/topol.top -o /data/md.tpr
docker run -v $(pwd):/data grinn gmx mdrun -v -deffnm /data/md

# 6. Run gRINN analysis on your trajectory
docker run -v $(pwd):/data grinn workflow /data/em.gro /data/results --tpr /data/md.tpr --traj /data/md.xtc

# 7. Launch dashboard
docker run -p 8051:8051 -v $(pwd):/data grinn dashboard /data/results
```

#### Analyze Existing Trajectory
```bash
# If you already have GROMACS files
docker run -v $(pwd):/data grinn workflow /data/protein.pdb /data/results --tpr /data/system.tpr --traj /data/trajectory.xtc

# Launch dashboard
docker run -p 8051:8051 -v $(pwd):/data grinn dashboard /data/results
```

**Note:** All force field files and MD parameters are included in the Docker container.

### Using Conda
```bash
# Setup environment
conda create -n grinn python=3.10
conda activate grinn
conda install -c conda-forge -c bioconda prody numpy scipy pandas mdtraj networkx tqdm pdbfixer openmm panedr gromacswrapper pyprind dash dash-bootstrap-components plotly
pip install dash-molstar

# Run analysis
python grinn_workflow.py protein.pdb results/ --top protein.top --traj trajectory.xtc

# Launch dashboard
python gRINN_Dashboard/grinn_dashboard.py results/
```

## 🎯 Key Features

### Core Functionality
- **Residue Interaction Energy Calculation**: Compute pairwise amino acid non-bonded interaction energies from GROMACS MD trajectories
- **Protein Energy Network (PEN) Construction**: Build energy-based networks with customizable cutoffs and analyze network properties
- **Betweenness Centrality Analysis**: Identify key residues in protein communication pathways
- **Multi-threading Support**: Parallel processing for efficient large-scale calculations
- **Flexible Residue Selection**: Custom source and target residue selections using ProDy syntax

### Advanced Features
- **Memory-Efficient Processing**: Optimized algorithms for handling large trajectories
- **Frame Skipping**: Analyze trajectories with configurable frame intervals
- **Comprehensive Validation**: Input validation and GROMACS compatibility testing
- **Automated Workflows**: Complete pipeline from structure preparation to network analysis
- **Docker Support**: Containerized deployment for reproducible results
- **Interactive Dashboard**: Web-based visualization and analysis interface with 3D molecular viewer

### Output Formats
- **Energy Matrices**: CSV files with interaction energies for all residue pairs
- **Network Files**: GML format networks for visualization and further analysis
- **Comprehensive Reports**: JSON and text summaries of workflow results
- **Interactive Visualization**: Web dashboard for real-time analysis and exploration
- **Visualization Ready**: Compatible with network analysis tools like Cytoscape

## 📦 Installation & Usage

### Option 1: Docker (Recommended)

The easiest way to run gRINN is using Docker. The container includes all dependencies and test data.

#### For Command Line Analysis:
```bash
# Build the Docker image
docker build -t grinn .

# Run with your data (mount your data directory)
docker run -v /path/to/your/data:/data grinn \
  /data/your_structure.pdb \
  /data/output_folder \
  --top /data/topology.top \
  --traj /data/trajectory.xtc \
  --nt 4 \
  --create_pen \
  --pen_cutoffs 1.0 2.0
```

#### For Interactive Dashboard:
```bash
# Build the Docker image (if not already built)
docker build -t grinn .

# Run the dashboard with your results
docker run -p 8051:8051 -v /path/to/your/results:/data grinn dashboard /data

# Run the dashboard with test data
docker run -p 8051:8051 grinn dashboard test

# Access the dashboard at http://localhost:8051
```

### Option 2: Conda Installation (Recommended for Local Development)

#### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual)
- GROMACS 2024.1 or later (can be installed via conda)

#### Installation Steps
```bash
# Option 1: Automated setup (recommended)
./setup_conda.sh

# Option 2: Use the provided environment file
conda env create -f environment.yml
conda activate grinn

# Option 3: Manual installation
conda create -n grinn python=3.10
conda activate grinn

# Install core dependencies
conda install -c conda-forge -c bioconda \
    prody numpy scipy pandas mdtraj networkx tqdm \
    pdbfixer openmm panedr gromacswrapper pyprind

# Install GROMACS (optional - you can also install from source)
conda install -c conda-forge -c bioconda gromacs

# For dashboard functionality, install additional packages
conda install -c conda-forge -c plotly \
    dash dash-bootstrap-components plotly pandas \
    networkx numpy

# Install dash-molstar for 3D visualization
pip install dash-molstar
```

#### Verify Installation
```bash
# Run the verification script to check all dependencies
./verify_installation.sh
```

#### Running gRINN Workflow
```bash
# Activate environment
conda activate grinn

# Basic usage
python grinn_workflow.py input.pdb output_folder

# With topology and trajectory
python grinn_workflow.py input.pdb output_folder \
  --top topology.top \
  --traj trajectory.xtc \
  --nt 8

# Full workflow with PEN analysis
python grinn_workflow.py input.pdb output_folder \
  --top topology.top \
  --traj trajectory.xtc \
  --nt 8 \
  --create_pen \
  --pen_cutoffs 0.5 1.0 2.0 \
  --pen_include_covalents True False
```

#### Running Interactive Dashboard
```bash
# Navigate to the dashboard directory
cd gRINN_Dashboard

# Run with your results
python grinn_dashboard.py /path/to/your/results

# Run with test data
python grinn_dashboard.py test

# Access at http://localhost:8051
```

### Option 3: Manual Installation

#### Prerequisites
- Python 3.10+
- GROMACS 2024.1 or later
- Required Python packages (see requirements below)

#### Required Python Packages
```bash
# Install via pip
pip install prody numpy scipy pandas mdtraj networkx tqdm \
           pdbfixer openmm panedr gromacswrapper pyprind \
           dash dash-bootstrap-components plotly dash-molstar
```

## 🎯 Interactive Dashboard

The gRINN Dashboard provides a web-based interface for exploring and visualizing interaction energy results. It features:

### Dashboard Features
- **3D Molecular Visualization**: Interactive protein structure viewer with trajectory support
- **Pairwise Energy Analysis**: Select residue pairs and visualize their interaction energies across frames
- **Energy Matrix Heatmaps**: Interactive heatmaps showing interaction patterns between all residue pairs
- **Network Analysis**: Protein Energy Network visualization with centrality metrics
- **Multi-Energy Support**: Analyze total, electrostatic, and van der Waals energies separately
- **Real-time Interaction**: Synchronized 3D viewer with energy plots and network analysis

### Dashboard Usage
```bash
# Run dashboard with your gRINN results
python gRINN_Dashboard/grinn_dashboard.py /path/to/results

# Run with test data
python gRINN_Dashboard/grinn_dashboard.py test

# Access at http://localhost:8051
```

### Dashboard Input Files
The dashboard expects these files in your results directory:
- `system_dry.pdb` - Protein structure (required)
- `energies_intEnTotal.csv` - Total interaction energies (required)
- `energies_intEnVdW.csv` - Van der Waals energies (required)
- `energies_intEnElec.csv` - Electrostatic energies (required)
- `traj_dry.xtc` - Trajectory file (optional, for dynamic visualization)

### Dashboard Tabs
1. **🔗 Pairwise Energies**: Select two residues and analyze their interaction energy over time
2. **🔥 Interaction Energy Matrix**: Visualize all pairwise interactions as interactive heatmaps
3. **🕸️ Network Analysis**: Construct and analyze Protein Energy Networks with centrality metrics

## 🛠️ Command Line Options

### Required Arguments
- `pdb_file`: Input PDB structure file
- `out_folder`: Output directory for results

### Optional Arguments
- `--top`: Topology file (.top)
- `--traj`: Trajectory file (.xtc, .trr, .dcd)
- `--nt`: Number of threads (default: 1)
- `--initpairfiltercutoff`: Distance cutoff for initial filtering (default: 10.0 Å)
- `--source_sel`: Source residue selection (ProDy syntax)
- `--target_sel`: Target residue selection (ProDy syntax)
- `--skip`: Frame skipping interval (default: 1)

### PEN Analysis Options
- `--create_pen`: Enable Protein Energy Network construction
- `--pen_cutoffs`: Energy cutoffs for network construction (default: [1.0])
- `--pen_include_covalents`: Include covalent bonds (default: [True, False])

### Advanced Options
- `--ff_folder`: Custom force field directory
- `--include_files`: Additional files to include
- `--gpu`: Enable GPU acceleration for GROMACS
- `--solvate`: Perform solvation
- `--npt`: Run NPT equilibration
- `--nointeraction`: Skip interaction energy calculation
- `--test-only`: Validate inputs without running workflow

## 📊 Example Workflows

### Basic Interaction Energy Analysis
```bash
# Calculate interaction energies for a protein
python grinn_workflow.py protein.pdb results/ \
  --top protein.top \
  --traj md_trajectory.xtc \
  --nt 4 \
  --source_sel "protein" \
  --target_sel "protein"
```

### Protein-Ligand Interaction Analysis
```bash
# Analyze protein-ligand interactions
python grinn_workflow.py complex.pdb results/ \
  --top complex.top \
  --traj md_trajectory.xtc \
  --nt 8 \
  --source_sel "protein" \
  --target_sel "resname LIG" \
  --create_pen \
  --pen_cutoffs 1.0 2.0
```

### Large-Scale Analysis with Frame Skipping
```bash
# Analyze every 10th frame for efficiency
python grinn_workflow.py protein.pdb results/ \
  --top protein.top \
  --traj long_trajectory.xtc \
  --nt 16 \
  --skip 10 \
  --create_pen \
  --pen_cutoffs 0.5 1.0 1.5 2.0
```

### Complete Workflow with Dashboard Visualization

#### Using Docker (Recommended):
```bash
# 1. Build the Docker image
docker build -t grinn .

# 2. Run gRINN analysis with your data
docker run -v /path/to/your/data:/data grinn \
  /data/protein.pdb \
  /data/results \
  --top /data/protein.top \
  --traj /data/trajectory.xtc \
  --nt 8 \
  --create_pen \
  --pen_cutoffs 1.0 2.0

# 3. Launch interactive dashboard
docker run -p 8051:8051 -v /path/to/your/data:/data grinn dashboard /data/results

# 4. Open http://localhost:8051 in your browser
```

#### Using Conda:
```bash
# 1. Run gRINN analysis
conda activate grinn
python grinn_workflow.py protein.pdb results/ \
  --top protein.top \
  --traj trajectory.xtc \
  --nt 8 \
  --create_pen \
  --pen_cutoffs 1.0 2.0

# 2. Launch interactive dashboard
python gRINN_Dashboard/grinn_dashboard.py results/

# 3. Open http://localhost:8051 in your browser
```

## 📁 Output Files

### Energy Analysis
- `energies_intEnTotal.csv`: Total interaction energies for all residue pairs
- `energies_intEnElec.csv`: Electrostatic interaction energies
- `energies_intEnVdW.csv`: Van der Waals interaction energies
- `energies_*.pickle`: Pickled energy dictionaries for programmatic access

### Structure and Trajectory Files
- `system_dry.pdb`: Processed protein structure (dashboard compatible)
- `traj_dry.xtc`: Dry trajectory (dashboard compatible)
- `topol_dry.top`: GROMACS topology file

### Network Analysis (if `--create_pen` enabled)
- `pen_*.gml`: Protein Energy Network files in GML format
- `pen_betweenness_centralities.csv`: Betweenness centrality values for all residues

### Reports and Logs
- `grinn_workflow_summary.json`: Comprehensive workflow summary
- `grinn_workflow_summary.txt`: Human-readable summary
- `calc.log`: Detailed calculation log

### Setup and Configuration Files
- `environment.yml`: Conda environment specification
- `setup_conda.sh`: Automated conda setup script
- `verify_installation.sh`: Installation verification script

### Dashboard Visualization
All energy CSV files and structure files are directly compatible with the interactive dashboard for real-time analysis and exploration.

## 🔬 Scientific Background

gRINN implements the methodology described in:

**Serçinoğlu, O., & Ozbek, P. (2018).** gRINN: a tool for calculation of residue interaction energies and protein energy network analysis of molecular dynamics simulations. *Nucleic Acids Research*, 46(W1), W554-W562. [https://doi.org/10.1093/nar/gky381](https://doi.org/10.1093/nar/gky381)

### Key Concepts
- **Residue Interaction Energies**: Non-bonded (electrostatic + van der Waals) energies between residue pairs
- **Protein Energy Networks**: Graph representations where nodes are residues and edges represent significant interactions
- **Betweenness Centrality**: Measure of a residue's importance in protein communication pathways

## 🧪 Testing & Validation

### Input Validation
```bash
# Test input compatibility without running the full workflow
python grinn_workflow.py protein.pdb test_output/ \
  --top protein.top \
  --traj trajectory.xtc \
  --test-only
```

### Container Testing
The Docker image includes comprehensive tests that run during build:
```bash
# Test with included sample data
docker run grinn bash -c "conda run -n grinn-env bash ./test.sh"
```

## 🔧 Advanced Configuration

### Custom Force Fields
```bash
# Use custom force field
python grinn_workflow.py protein.pdb results/ \
  --ff_folder /path/to/custom_ff/ \
  --include_files custom_params.itp
```

### Selection Syntax
gRINN uses ProDy selection syntax for flexible residue selection:
```bash
# Select specific chains
--source_sel "chain A"
--target_sel "chain B"

# Select by residue type
--source_sel "protein"
--target_sel "resname LIG"

# Complex selections
--source_sel "chain A and resid 1:100"
--target_sel "chain B and resname ARG LYS"
```

## 🐛 Troubleshooting

### Common Issues
1. **GROMACS not found**: Ensure GROMACS is installed and `gmx` is in PATH
2. **Memory issues**: Use `--skip` to reduce trajectory size or increase available RAM
3. **Topology errors**: Verify topology file matches the PDB structure
4. **Missing files**: Use `--test-only` to validate inputs before running
5. **Dashboard not accessible**: Ensure port 8051 is not blocked and Docker port mapping is correct

### Docker-Specific Issues
- **Port conflicts**: If port 8051 is in use, try `-p 8052:8051` and access via `http://localhost:8052`
- **File permissions**: Use `docker run --user $(id -u):$(id -g)` to match host permissions
- **Volume mounting**: Ensure your data directory exists and has proper permissions

### Performance Tips
- Use `--nt` to leverage multiple CPU cores
- Enable `--skip` for large trajectories
- Use Docker for consistent performance across systems
- Monitor disk space for large calculations

## 📚 Documentation

A detail documentation for gRINN is currently under development.

## 🤝 Contributing

This repository contains the next-generation version of gRINN. For bug reports, feature requests, or contributions, please contact the developers.

## 📄 License

This software is distributed under a custom license. See the [original gRINN documentation](https://grinn.readthedocs.io/en/latest/license.html) for details.

## 📞 Contact

For questions, issues, or collaborations, please refer to the [original gRINN contact information](https://grinn.readthedocs.io/en/latest/contact.html).

---

*This is the optimized, containerized version of gRINN, building upon the original work by Onur Serçinoğlu and Pemra Ozbek.*
