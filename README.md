# gRINN: Protein Energy Networks from Molecular Dynamics

[![License](https://img.shields.io/badge/License-Custom-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-brightgreen.svg)](https://python.org)
[![GROMACS](https://img.shields.io/badge/GROMACS-2020.7--2025.2-orange.svg)](https://www.gromacs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

> **🚧 Development Notice**: This tool is currently under active development. We welcome users to test gRINN and report any issues they encounter in the [Issues](../../issues) section. Your feedback helps us improve!

**gRINN** (get Residue Interaction Energies and Networks) is a computational tool for **residue interaction energy-based analysis** of protein molecular dynamics (MD) simulation trajectories. This is the **next-generation version** of gRINN, featuring significant performance improvements, enhanced workflow capabilities, and a unified containerized deployment system.

## 🎯 Key Features

- **Residue Interaction Energy Calculation**: Compute pairwise residue (not only amino acids!) interaction energies from GROMACS MD trajectories.
- **Protein Energy Networks**: Build and analyze energy-based network representations of protein structural ensembles.
- **Interactive Dashboard**: Web-based visualization with 3D molecular viewer and energy analysis
- **Multi-Version GROMACS Support**: Compatible with GROMACS versions 2020.7 through 2025.2

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

#### Linux/Mac Build
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

#### Windows Build
For Windows users, use PowerShell or Command Prompt:
```powershell
# 1. Build gRINN container (replace <VERSION> with desired GROMACS version)
docker build --build-arg GROMACS_VERSION=2024.1 -t grinn:gromacs-2024.1 .

# 2. Run analysis (adjust paths using Windows format)
docker run -v C:\path\to\your\data:/data grinn:gromacs-2024.1 workflow /data/protein.pdb /data/results --top /data/topology.top --traj /data/trajectory.xtc

# 3. Launch dashboard
docker run -p 8051:8051 -v C:\path\to\your\data:/data grinn:gromacs-2024.1 dashboard /data/results
# Open http://localhost:8051 in your browser
```

#### Docker Cleanup
After building, you may want to clean up intermediate build artifacts:
```bash
# Remove dangling images (saves disk space)
docker image prune -f

# Remove all stopped containers
docker container prune -f

# Complete cleanup (removes all unused Docker objects)
docker system prune -a -f
```

### GROMACS Version Support

#### Using Build Script (Linux/Mac)
```bash
./build-grinn.sh 2025.2  # Latest features
./build-grinn.sh 2024.1  # Current stable  
./build-grinn.sh 2020.7  # Legacy support
```

#### Manual Docker Build (All Platforms)
```bash
# Modern GROMACS versions (2021.7 - 2025.2)
docker build --build-arg GROMACS_VERSION=2024.1 -t grinn:gromacs-2024.1 .

# Legacy GROMACS 2020.7 (requires special Dockerfile)
docker build -f Dockerfile.gromacs-2020.7 -t grinn:gromacs-2020.7 .
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

### Launch Dashboard

#### Docker (Recommended)
```bash
# Linux/Mac
docker run -p 8051:8051 -v /path/to/results:/data grinn:gromacs-2024.1 dashboard /data

# Windows
docker run -p 8051:8051 -v C:\path\to\results:/data grinn:gromacs-2024.1 dashboard /data

# Open http://localhost:8051
```

#### Conda
```bash
python gRINN_Dashboard/grinn_dashboard.py results/
# Open http://localhost:8051
```

## 🛠️ Command Line Options

### Docker Usage
```bash
# Linux/Mac
docker run -v /path/to/data:/data grinn:gromacs-2024.1 workflow \
  /data/protein.pdb /data/results --top /data/topology.top --traj /data/trajectory.xtc

# Windows
docker run -v C:\path\to\data:/data grinn:gromacs-2024.1 workflow /data/protein.pdb /data/results --top /data/topology.top --traj /data/trajectory.xtc
```

### Conda Usage
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

### Docker Workflows (Recommended)

#### Basic Analysis
```bash
# Linux/Mac
docker run -v /path/to/data:/data grinn:gromacs-2024.1 workflow \
  /data/protein.pdb /data/results --top /data/protein.top --traj /data/trajectory.xtc

# Windows
docker run -v C:\path\to\data:/data grinn:gromacs-2024.1 workflow /data/protein.pdb /data/results --top /data/protein.top --traj /data/trajectory.xtc
```

#### Large Trajectory Analysis (with frame skipping)
```bash
# Linux/Mac
docker run -v /path/to/data:/data grinn:gromacs-2024.1 workflow \
  /data/protein.pdb /data/results \
  --top /data/protein.top --traj /data/trajectory.xtc \
  --skip 10 --create_pen

# Windows
docker run -v C:\path\to\data:/data grinn:gromacs-2024.1 workflow /data/protein.pdb /data/results --top /data/protein.top --traj /data/trajectory.xtc --skip 10 --create_pen
```

### Conda Workflows (Local Development)

#### Basic Analysis
```bash
python grinn_workflow.py protein.pdb results/ --top protein.top --traj trajectory.xtc
```

#### Large Trajectory (with frame skipping)
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

**Technical Note**: gRINN is essentially a sophisticated wrapper around `gmx mdrun -rerun` that automates the calculation of pairwise residue interaction energies and provides tools for network analysis and visualization.

##  Troubleshooting

### Common Issues
- **Port 8051 in use**: Try `-p 8052:8051` and access via `http://localhost:8052`
- **Large trajectories**: Use `--skip 10` to analyze every 10th frame
- **Permission errors (Linux)**: Make sure you're in the docker group (see installation)
- **Windows path issues**: Use forward slashes in container paths: `/data/file.pdb` not `\data\file.pdb`
- **Docker build fails**: Try cleaning up first: `docker system prune -a -f`

### Windows-Specific Tips
- Use **PowerShell** or **Command Prompt** for Docker commands
- Mount Windows drives: `-v C:\Users\YourName\data:/data` 
- For WSL2 users: Access files via `/mnt/c/Users/YourName/data`
- If path contains spaces, wrap in quotes: `-v "C:\My Data":/data`

### Get Help
```bash
# Docker help
docker run grinn:gromacs-2024.1 help           # Container usage
docker run grinn:gromacs-2024.1 workflow --help  # Workflow options

# Conda help  
python grinn_workflow.py --help                # Workflow options

# Build help (Linux/Mac)
./build-grinn.sh --help                        # Build script options
```

## 🤝 Contributing

For bug reports, feature requests, or contributions, please contact the developers.

## 📄 License

See the [original gRINN documentation](https://grinn.readthedocs.io/en/latest/license.html) for license details.

---

*This is the optimized, containerized version of gRINN, building upon the original work by Onur Serçinoğlu and Pemra Ozbek.*
