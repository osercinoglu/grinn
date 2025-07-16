# gRINN: Protein Energy Networks from Molecular Dynamics

[![License](https://img.shields.io/badge/License-Custom-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-brightgreen.svg)](https://python.org)
[![GROMACS](https://img.shields.io/badge/GROMACS-2024.1-orange.svg)](https://www.gromacs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

**gRINN** (get Residue Interaction Energies and Networks) is a computational tool for **residue interaction energy-based analysis** of protein molecular dynamics (MD) simulation trajectories. This is the **next-generation version** of gRINN, featuring significant performance improvements, enhanced workflow capabilities, and containerized deployment.

## 🚀 Key Features

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

### Output Formats
- **Energy Matrices**: CSV files with interaction energies for all residue pairs
- **Network Files**: GML format networks for visualization and further analysis
- **Comprehensive Reports**: JSON and text summaries of workflow results
- **Visualization Ready**: Compatible with network analysis tools like Cytoscape

## 📦 Installation & Usage

### Option 1: Docker (Recommended)

The easiest way to run gRINN is using Docker. The container includes all dependencies and test data.

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

### Option 2: Local Installation

#### Prerequisites
- Python 3.10+
- GROMACS 2024.1 or later
- Required Python packages (see requirements below)

#### Required Python Packages
```bash
# Install via conda (recommended)
conda create -n grinn python=3.10
conda activate grinn
conda install -c conda-forge -c bioconda \
  prody numpy scipy pandas mdtraj networkx tqdm \
  pdbfixer openmm panedr gromacswrapper pyprind
```

#### Running gRINN
```bash
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

## 📁 Output Files

### Energy Analysis
- `energies_intEnTotal.csv`: Total interaction energies for all residue pairs
- `energies_intEnElec.csv`: Electrostatic interaction energies
- `energies_intEnVdW.csv`: Van der Waals interaction energies
- `energies_*.pickle`: Pickled energy dictionaries for programmatic access

### Network Analysis (if `--create_pen` enabled)
- `pen_*.gml`: Protein Energy Network files in GML format
- `pen_betweenness_centralities.csv`: Betweenness centrality values for all residues

### Reports and Logs
- `grinn_workflow_summary.json`: Comprehensive workflow summary
- `grinn_workflow_summary.txt`: Human-readable summary
- `calc.log`: Detailed calculation log

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
