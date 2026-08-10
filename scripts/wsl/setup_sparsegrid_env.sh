#!/usr/bin/env bash
# Create (or repair) the WSL "sparsegrid" conda env used for Clenshaw-Curtis
# sparse-grid quadrature (Tasmanian).
#
# Tasmanian has no conda-forge package, so it is built from source via pip.
# Its PyPI build (scikit-build) bakes an ABSOLUTE hardcoded path
# ("/root/.local/lib/lib*.so") into TasmanianConfig.py for the shared
# libraries, regardless of where they actually get installed (here: the
# sparsegrid conda env's own lib/ dir). Without the symlink step below,
# `import Tasmanian` fails with OSError: cannot open shared object file.
# Re-run this script any time the sparsegrid env is recreated from scratch.
set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"

if ! conda env list | grep -q '^sparsegrid '; then
    conda create -y -n sparsegrid -c conda-forge python=3.10 numpy scipy pytest
fi

conda activate sparsegrid

if ! python -c "import Tasmanian" 2>/dev/null; then
    conda install -y -c conda-forge cmake make cxx-compiler
    pip install Tasmanian --no-cache-dir
fi

mkdir -p /root/.local/lib
ln -sf "$CONDA_PREFIX/lib"/libtasmanian*.so* /root/.local/lib/

python -c "import Tasmanian; print('Tasmanian OK:', Tasmanian.__version__ if hasattr(Tasmanian, '__version__') else 'imported')"
