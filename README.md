# gRINN: Protein Energy Networks from Molecular Dynamics

[![License](https://img.shields.io/badge/License-Custom-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-brightgreen.svg)](https://python.org)
[![GROMACS](https://img.shields.io/badge/GROMACS-2020.7--2025.2-orange.svg)](https://www.gromacs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

> **🚧 Development Notice**: This tool is currently under active development. We welcome users to test gRINN and report any issues they encounter in the [Issues](../../issues) section.

**gRINN** (get Residue Interaction Energies and Networks) calculates pairwise residue interaction energies from GROMACS MD trajectories or conformational ensembles and builds energy-based protein networks. Includes an interactive web dashboard for 3D visualization and analysis.

## Input Modes

**Trajectory Mode** - Analyze existing MD simulation:
- Structure: PDB (`.pdb`)
- Trajectory: XTC (`.xtc`)
- Topology: TOP (`.top`)

**Ensemble Mode** - Analyze conformational ensemble:
- Multi-model PDB file (multiple MODEL entries)
- Topology generated automatically

## Quick Start

### Docker (Recommended)

**Build:**
```bash
./build-grinn.sh 2024.1  # Linux/Mac
docker build --build-arg GROMACS_VERSION=2024.1 -t grinn:gromacs-2024.1 .  # Windows
```

**Run Trajectory Mode:**
```bash
docker run -v /path/to/data:/data grinn:gromacs-2024.1 workflow \
  /data/protein.pdb /data/results --traj /data/trajectory.xtc --top /data/topology.top
```

**Run Ensemble Mode:**
```bash
docker run -v /path/to/data:/data grinn:gromacs-2024.1 workflow \
  /data/ensemble.pdb /data/results --ensemble_mode
```

**Launch Dashboard:**
```bash
docker run -p 8051:8051 -v /path/to/data:/data grinn:gromacs-2024.1 dashboard /data/results
# Open http://localhost:8051
```

### Conda Alternative

```bash
conda env create -f environment.yml
conda activate grinn

# Trajectory mode
python grinn_workflow.py protein.pdb results/ --traj trajectory.xtc --top topology.top

# Ensemble mode
python grinn_workflow.py ensemble.pdb results/ --ensemble_mode

# Dashboard
python gRINN_Dashboard/grinn_dashboard.py results/
```

## Common Options

- `--ensemble_mode` - Process multi-model PDB
- `--create_pen` - Generate Protein Energy Networks
- `--skip N` - Analyze every N-th frame (for large trajectories)
- `--nt N` - Number of threads (auto-detects by default)
- `--force_field` - Force field for topology (default: amber99sb-ildn)
- `--test-only` - Validate inputs without running analysis

## Output Files

- `energies_intEnTotal.csv` - Total interaction energies
- `energies_intEnElec.csv` - Electrostatic interactions  
- `energies_intEnVdW.csv` - Van der Waals interactions
- `pen_*.gml` - Protein Energy Network files (with `--create_pen`)
- `system_dry.pdb` - Processed structure

## Troubleshooting

**Permission denied (Linux):**
```bash
sudo usermod -aG docker $USER  # Log out and back in
sudo systemctl start docker
```

**Docker not running:**
- Linux: `sudo systemctl start docker`
- Windows/Mac: Start Docker Desktop

**Out of memory / large trajectories:**
```bash
docker run -v /path/to/data:/data grinn:gromacs-2024.1 workflow \
  /data/protein.pdb /data/results --traj /data/trajectory.xtc --top /data/topology.top --skip 10
```

**Windows paths:**
```powershell
docker run -v C:/Users/YourName/data:/data grinn:gromacs-2024.1 workflow ...
# With spaces: -v "C:/My Data":/data
```

**Port 8051 in use:**
```bash
docker run -p 8052:8051 -v /path/to/data:/data grinn:gromacs-2024.1 dashboard /data/results
# Access via http://localhost:8052
```

**Input requirements:**
- Structure: PDB format only
- Trajectory: XTC format (trajectory mode)
- Topology: TOP format (trajectory mode)
- Ensemble: Multi-model PDB with MODEL/ENDMDL entries

Report issues at [Issues](../../issues).

## Contributing

See [CONTRIBUTING.md](../grinn-ismb-2025/CONTRIBUTING.md) for guidelines.

## License

Copyright 2024-2025. Available for academic and research use only. See [LICENSE](LICENSE) for details.
