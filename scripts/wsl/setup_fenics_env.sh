#!/usr/bin/env bash
# Create (or repair) the WSL "fenics2019" conda env used for the mixed FEM
# diffusion/NSB solvers (src/ol_reproduction/pde/**/fenics*_solver.py).
#
# Note: mshr (conda-forge build mshr=2019.1.0) has a known pybind11 ABI
# incompatibility with fenics=2019.1.0 in this environment
# ("ImportError: generic_type: type CSGGeometry referenced unknown base
# type dolfin::Variable" -- reproducible via scripts/wsl/test_mshr_import.py,
# not fixed by reinstalling via conda or mamba). The diffusion solver
# (pde/diffusion/fenics_mixed_solver.py) therefore falls back to a
# structured UnitSquareMesh when mshr isn't importable -- this script does
# not attempt to fix mshr, only documents the gap.
set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"

if ! conda env list | grep -q '^fenics2019 '; then
    conda create -y -n fenics2019 -c conda-forge python=3.7 fenics=2019.1.0 mshr numpy scipy matplotlib pytest
fi

conda activate fenics2019

python -c "import dolfin; print('dolfin OK', dolfin.__version__)"
python -c "import scipy; print('scipy OK', scipy.__version__)"
