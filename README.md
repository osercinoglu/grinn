# gRINN: Protein Energy Networks from Molecular Dynamics

[![License](https://img.shields.io/badge/License-Custom-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-brightgreen.svg)](https://python.org)
[![GROMACS](https://img.shields.io/badge/GROMACS-2020.7--2025.2-orange.svg)](https://www.gromacs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

**gRINN** (get Residue Interaction Energies and Networks) is a computational tool for **residue interaction energy-based analysis** of protein molecular dynamics (MD) simulation trajectories. This is the **next-generation version** of gRINN, featuring significant performance improvements, enhanced workflow capabilities, and a unified containerized deployment system.

## 🎯 Key Features

- **Residue Interaction Energy Calculation**: Compute pairwise amino acid interaction energies from GROMACS MD trajectories
- **Protein Energy Networks**: Build and analyze energy-based network representations of protein communication
- **Interactive Dashboard**: Web-based visualization with 3D molecular viewer and energy analysis
- **Multi-Version GROMACS Support**: Compatible with GROMACS versions 2020.7 through 2025.2
- **Automated Workflows**: Complete pipeline from structure preparation to network analysis

## 📦 Installation & Usage

### Prerequisites

First, install Docker on your system:
- **Windows/Mac**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install docker.io`
- **CentOS/RHEL**: `sudo yum install docker`

**Linux users**: Add yourself to the docker group to run without sudo:
```bash
sudo usermod -aG docker $USER
# Log out and log back in, then test:
docker run hello-world
```

### Quick Start (Docker - Recommended)

```bash
# 1. Build gRINN container
./build-grinn.sh 2024.1

# 2. Run analysis on your trajectory
docker run -v /path/to/your/data:/data grinn:gromacs-2024.1 workflow \
  /data/protein.pdb /data/results \
  --top /data/topology.top \
  --traj /data/trajectory.xtc

# 3. Launch interactive dashboard
docker run -p 8051:8051 -v /path/to/your/data:/data grinn:gromacs-2024.1 dashboard /data/results
# Open http://localhost:8051 in your browser
```

### GROMACS Version Support

Build for any GROMACS version (2020.7 - 2025.2):
```bash
./build-grinn.sh 2025.2  # Latest features
./build-grinn.sh 2024.1  # Current stable  
./build-grinn.sh 2020.7  # Legacy support
```

### Alternative: Conda Installation

For local development or if you prefer conda:
```bash
# Setup environment
conda env create -f environment.yml
conda activate grinn

# Run analysis
python grinn_workflow.py protein.pdb results/ --top protein.top --traj trajectory.xtc

# Launch dashboard
python gRINN_Dashboard/grinn_dashboard.py results/
```

## 🎯 Interactive Dashboard

Web-based interface for exploring results:
- **3D Molecular Visualization**: Interactive protein structure viewer
- **Energy Analysis**: Pairwise interactions and heatmaps  
- **Network Analysis**: Protein Energy Network visualization

```bash
# Run dashboard with your results
docker run -p 8051:8051 -v /path/to/results:/data grinn:gromacs-2024.1 dashboard /data
# Open http://localhost:8051
```

## 🛠️ Command Line Options

### Basic Usage
```bash
python grinn_workflow.py protein.pdb results/ --top topology.top --traj trajectory.xtc
```

### Key Options
- `--nt`: Number of threads (auto-detects CPU cores by default)
- `--create_pen`: Enable Protein Energy Network analysis
- `--pen_cutoffs`: Energy cutoffs for networks (default: 1.0)
- `--skip`: Analyze every N-th frame for large trajectories
- `--source_sel / --target_sel`: Custom residue selections (ProDy syntax)

> **Note**: gRINN automatically uses all available CPU cores. Override with `--nt <number>` if needed.

## 📊 Example Workflows

### Basic Analysis
```bash
python grinn_workflow.py protein.pdb results/ --top protein.top --traj trajectory.xtc
```

### Protein-Ligand Interactions
```bash
python grinn_workflow.py complex.pdb results/ \
  --top complex.top --traj trajectory.xtc \
  --source_sel "protein" --target_sel "resname LIG" \
  --create_pen --pen_cutoffs 1.0 2.0
```

### Large Trajectory (with frame skipping)
```bash
python grinn_workflow.py protein.pdb results/ \
  --top protein.top --traj trajectory.xtc \
  --skip 10 --create_pen
```

## 📁 Output Files

- `energies_intEnTotal.csv` - Total interaction energies between residue pairs
- `energies_intEnElec.csv` - Electrostatic interactions  
- `energies_intEnVdW.csv` - Van der Waals interactions
- `pen_*.gml` - Protein Energy Network files (if `--create_pen` enabled)
- `system_dry.pdb` - Processed structure (dashboard compatible)

## 🔬 Scientific Background

gRINN implements the methodology from:

**Serçinoğlu, O., & Ozbek, P. (2018).** gRINN: a tool for calculation of residue interaction energies and protein energy network analysis of molecular dynamics simulations. *Nucleic Acids Research*, 46(W1), W554-W562. [https://doi.org/10.1093/nar/gky381](https://doi.org/10.1093/nar/gky381)

## 📚 Documentation

For advanced usage and build system details, see [BUILD-SYSTEM.md](BUILD-SYSTEM.md).

## 🐛 Troubleshooting

### Common Issues
- **Port 8051 in use**: Try `-p 8052:8051` and access via `http://localhost:8052`
- **Large trajectories**: Use `--skip 10` to analyze every 10th frame
- **Permission errors (Linux)**: Make sure you're in the docker group (see installation)

### Get Help
```bash
./build-grinn.sh --help                    # Build script options
python grinn_workflow.py --help           # Workflow options
```

## 📚 Documentation

For advanced usage and build system details, see [BUILD-SYSTEM.md](BUILD-SYSTEM.md).

## 🤝 Contributing

For bug reports, feature requests, or contributions, please contact the developers.

## 📄 License

See the [original gRINN documentation](https://grinn.readthedocs.io/en/latest/license.html) for license details.

---

*This is the optimized, containerized version of gRINN, building upon the original work by Onur Serçinoğlu and Pemra Ozbek.*
