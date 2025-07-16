#!/bin/bash
# gRINN Setup Script for Conda Installation

set -e  # Exit on any error

echo "================================================"
echo "gRINN Installation Script"
echo "================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Check if environment file exists
if [ ! -f "environment.yml" ]; then
    echo "❌ environment.yml not found. Please run this script from the gRINN directory."
    exit 1
fi

echo "✅ Environment file found"

# Remove existing environment if it exists
if conda env list | grep -q "^grinn "; then
    echo "⚠️  Removing existing 'grinn' environment..."
    conda env remove -n grinn -y
fi

# Create new environment
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate environment and verify
echo "🔍 Verifying installation..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate grinn

# Check Python version
python_version=$(python --version 2>&1)
echo "✅ Python version: $python_version"

# Check key packages
echo "🔍 Checking key packages..."
packages=("prody" "numpy" "scipy" "pandas" "mdtraj" "networkx" "dash" "plotly")

for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✅ $package"
    else
        echo "❌ $package (failed to import)"
    fi
done

# Check GROMACS
echo "🔍 Checking GROMACS..."
if command -v gmx &> /dev/null; then
    echo "✅ GROMACS found: $(gmx --version 2>&1 | head -1)"
else
    echo "⚠️  GROMACS not found in PATH. You may need to install it separately."
fi

# Check gRINN files
echo "🔍 Checking gRINN files..."
if [ -f "grinn_workflow.py" ]; then
    echo "✅ grinn_workflow.py"
else
    echo "❌ grinn_workflow.py not found"
fi

if [ -f "gRINN_Dashboard/grinn_dashboard.py" ]; then
    echo "✅ gRINN_Dashboard/grinn_dashboard.py"
else
    echo "❌ gRINN_Dashboard/grinn_dashboard.py not found"
fi

echo "================================================"
echo "🎉 Installation complete!"
echo "================================================"
echo ""
echo "To use gRINN:"
echo "1. Activate the environment: conda activate grinn"
echo "2. Run workflow: python grinn_workflow.py input.pdb output_folder --help"
echo "3. Run dashboard: python gRINN_Dashboard/grinn_dashboard.py test"
echo ""
echo "For more information, see the README.md file."
