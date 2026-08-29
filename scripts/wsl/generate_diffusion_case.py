"""Phase 6/7: generate a full diffusion case dataset using the mixed FEM
solver + Clenshaw-Curtis sparse-grid test set.

Produces, under data/processed/<case_name>/:
    test.npz          -- shared sparse-grid test set: x, w (weights), y_u
    train_m{m}_seed{s}.npz  -- one file per (m, seed) combo: x, y_u
    mass_matrix.npz    -- FEM mass matrix for the u (DG0) space
    metadata.json       -- mesh/DOF/sparse-grid provenance

Usage (from repo root, inside the WSL 'fenics2019' conda env):
    python scripts/wsl/generate_diffusion_case.py \
        --case-name diffusion_affine_d4 --coefficient affine --dimension 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ol_reproduction.data.dataset_io import save_npz_dataset  # noqa: E402
from ol_reproduction.data.sampling import sample_uniform_parameters  # noqa: E402
from ol_reproduction.pde.diffusion.fenics_mixed_solver import (  # noqa: E402
    load_original_mesh,
    solve_diffusion_mixed_fenics,
)
from ol_reproduction.pde.mass_matrix import (  # noqa: E402
    assemble_mass_matrix_csr,
    save_mass_matrix_npz,
)

TRAIN_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
SEEDS = list(range(12))
FE_DEGREE = 1  # matches the authors' --FE_degree default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--coefficient", required=True, choices=["affine", "log"])
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--sparse-grid-level", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to data/processed/<case-name>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "data/processed" / args.case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading the authors' original mesh (data/original_mesh/poisson.xml)...")
    mesh = load_original_mesh()

    # Build one solve to get a handle on the DG1 u function space for the
    # mass matrix (solve_diffusion_mixed_fenics doesn't expose the space
    # object directly, so rebuild it identically here -- cheap, cached JIT).
    import dolfin as df

    u_element = df.FiniteElement("DG", mesh.ufl_cell(), FE_DEGREE)
    u_space = df.FunctionSpace(mesh, u_element)
    print("Assembling mass matrix...")
    mass_matrix = assemble_mass_matrix_csr(u_space)
    save_mass_matrix_npz(output_dir / "mass_matrix.npz", mass_matrix)

    from ol_reproduction.data.sparse_grid import (
        DEFAULT_LEVEL_BY_DIMENSION,
        load_precomputed_quadrature,
        validate_quadrature,
    )

    level = args.sparse_grid_level or DEFAULT_LEVEL_BY_DIMENSION[args.dimension]
    sparse_grid_path = repo_root / "data" / "sparse_grids" / f"d{args.dimension}_level{level}.npz"
    if not sparse_grid_path.exists():
        raise FileNotFoundError(
            f"{sparse_grid_path} not found. Generate it first (in the WSL "
            "'sparsegrid' conda env, which has Tasmanian): "
            f"python scripts/wsl/generate_sparse_grid_points.py "
            f"--dimension {args.dimension} --level {level}"
        )
    quadrature = load_precomputed_quadrature(sparse_grid_path)
    validate_quadrature(quadrature)
    print(f"Sparse grid: dimension={args.dimension} level={level} m_test={quadrature.points.shape[0]}")

    # --- Shared test set (solved once, reused by every m/trial) ---
    print(f"Solving {quadrature.points.shape[0]} test points...")
    start = time.perf_counter()
    test_solutions = []
    for i, parameter_vector in enumerate(quadrature.points):
        result = solve_diffusion_mixed_fenics(
            mesh=mesh,
            coefficient_name=args.coefficient,
            parameters=parameter_vector,
        )
        test_solutions.append(result.u_dofs.astype(np.float32))
        if (i + 1) % 200 == 0:
            print(f"  test point {i + 1}/{quadrature.points.shape[0]}")
    test_elapsed = time.perf_counter() - start
    print(f"Test set solved in {test_elapsed:.1f}s")

    save_npz_dataset(
        path=output_dir / "test.npz",
        arrays={
            "x": quadrature.points.astype(np.float32),
            "w": quadrature.weights.astype(np.float64),
            "y_u": np.stack(test_solutions, axis=0),
        },
    )

    # --- Training sets ---
    # Each seed solves only max(TRAIN_SIZES) points once; smaller m-value
    # files are prefix slices of that one solve, not separately re-solved.
    total_train_solves = len(SEEDS) * max(TRAIN_SIZES)
    print(f"Solving {total_train_solves} training points across {len(SEEDS)} seeds "
          f"({max(TRAIN_SIZES)} solves/seed, sliced into {len(TRAIN_SIZES)} m-value files each)...")
    start = time.perf_counter()
    solved_so_far = 0
    for seed in SEEDS:
        parameters = sample_uniform_parameters(
            num_samples=max(TRAIN_SIZES), dimension=args.dimension, seed=seed, dtype=np.float32
        )
        solutions = []
        for parameter_vector in parameters:
            result = solve_diffusion_mixed_fenics(
                mesh=mesh, coefficient_name=args.coefficient, parameters=parameter_vector,
            )
            solutions.append(result.u_dofs.astype(np.float32))
            solved_so_far += 1
            if solved_so_far % 500 == 0:
                elapsed = time.perf_counter() - start
                rate = solved_so_far / elapsed
                remaining = (total_train_solves - solved_so_far) / rate
                print(f"  {solved_so_far}/{total_train_solves} solves, "
                      f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

        solutions_array = np.stack(solutions, axis=0)
        for m in TRAIN_SIZES:
            save_npz_dataset(
                path=output_dir / f"train_m{m}_seed{seed}.npz",
                arrays={
                    "x": parameters[:m],
                    "y_u": solutions_array[:m],
                },
            )

    train_elapsed = time.perf_counter() - start
    print(f"Training sets solved in {train_elapsed:.1f}s")

    metadata = {
        "case_name": args.case_name,
        "coefficient": args.coefficient,
        "dimension": args.dimension,
        "mesh": {
            "method": "original_authors_mesh",
            "source": "data/original_mesh/poisson.xml",
            "fe_degree": FE_DEGREE,
            "num_cells": mesh.num_cells(),
            "h_min": mesh.hmin(),
            "h_max": mesh.hmax(),
            "num_dofs_u": u_space.dim(),
        },
        "sparse_grid": {
            "toolkit": "tasmanian",
            "rule": "clenshaw-curtis",
            "level": level,
            "m_test": int(quadrature.points.shape[0]),
        },
        "train_sizes": TRAIN_SIZES,
        "seeds": SEEDS,
        "timing": {
            "test_solve_seconds": test_elapsed,
            "train_solve_seconds": train_elapsed,
            "total_solves": total_train_solves + int(quadrature.points.shape[0]),
        },
        "note": (
            "Generated with the mixed FEM diffusion solver "
            "(pde/diffusion/fenics_mixed_solver.py), paper eq. B.9-B.12, "
            "BDM2 x DG1 elements on the paper authors' own mesh "
            "(data/original_mesh/poisson.xml)."
        ),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
