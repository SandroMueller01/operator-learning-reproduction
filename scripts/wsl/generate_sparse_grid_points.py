"""Generate and save Clenshaw-Curtis sparse-grid points/weights.

Runs in the WSL 'sparsegrid' conda env (has Tasmanian). Saves to
data/sparse_grids/d{dimension}_level{level}.npz so the 'fenics2019' env
(which does not have Tasmanian) can load the points without needing
Tasmanian itself -- see generate_diffusion_case.py / generate_nsb_case.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ol_reproduction.data.sparse_grid import (
    DEFAULT_LEVEL_BY_DIMENSION,
    build_clenshaw_curtis_quadrature,
    validate_quadrature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--level", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    level = args.level or DEFAULT_LEVEL_BY_DIMENSION[args.dimension]

    quadrature = build_clenshaw_curtis_quadrature(dimension=args.dimension, level=level)
    validate_quadrature(quadrature)

    output_dir = Path(__file__).resolve().parents[2] / "data" / "sparse_grids"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"d{args.dimension}_level{level}.npz"

    np.savez_compressed(
        output_path,
        points=quadrature.points,
        weights=quadrature.weights,
        dimension=args.dimension,
        level=level,
    )
    print(f"Saved {quadrature.points.shape[0]} points to {output_path}")


if __name__ == "__main__":
    main()
