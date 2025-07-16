#!/bin/bash
# gRINN Installation Verification Script

echo "================================================"
echo "gRINN Installation Verification"
echo "================================================"

# Check Python version
echo "Checking Python version..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python not found. Please install Python 3.10+"
    exit 1
fi

# Check if we're in a conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ Running in conda environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  Not in a conda environment. Consider using 'conda activate grinn'"
fi

# Check required packages
echo -e "\nChecking required packages..."
packages=("prody" "numpy" "scipy" "pandas" "mdtraj" "networkx" "tqdm" "dash" "plotly")

for package in "${packages[@]}"; do
    python -c "import $package" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ $package"
    else
        echo "❌ $package (missing)"
    fi
done

# Check GROMACS
echo -e "\nChecking GROMACS..."
which gmx >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ GROMACS found at $(which gmx)"
    gmx --version 2>&1 | head -1
else
    echo "❌ GROMACS not found in PATH"
fi

# Check gRINN files
echo -e "\nChecking gRINN files..."
if [ -f "grinn_workflow.py" ]; then
    echo "✅ grinn_workflow.py found"
else
    echo "❌ grinn_workflow.py not found"
fi

if [ -d "gRINN_Dashboard" ]; then
    echo "✅ gRINN_Dashboard directory found"
    if [ -f "gRINN_Dashboard/grinn_dashboard.py" ]; then
        echo "✅ grinn_dashboard.py found"
    else
        echo "❌ grinn_dashboard.py not found"
    fi
else
    echo "❌ gRINN_Dashboard directory not found"
fi

if [ -d "mdp_files" ]; then
    echo "✅ mdp_files directory found"
else
    echo "❌ mdp_files directory not found"
fi

echo -e "\n================================================"
echo "Installation verification complete!"
echo "================================================"

# Test basic functionality
echo -e "\nRunning basic functionality test..."
python -c "
import sys
sys.path.append('.')
try:
    from grinn_workflow import test_grinn_inputs
    print('✅ gRINN workflow can be imported')
except Exception as e:
    print(f'❌ gRINN workflow import failed: {e}')

try:
    import dash
    import plotly
    print('✅ Dashboard dependencies available')
except Exception as e:
    print(f'❌ Dashboard dependencies missing: {e}')
"

echo -e "\nTo run gRINN workflow:"
echo "python grinn_workflow.py input.pdb output_folder --help"
echo -e "\nTo run dashboard:"
echo "python gRINN_Dashboard/grinn_dashboard.py test"
