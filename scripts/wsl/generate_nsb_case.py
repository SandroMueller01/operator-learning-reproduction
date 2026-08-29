"""Phase 7: generate a full NSB case dataset using the nonlinear mixed FEM
solver + Clenshaw-Curtis sparse-grid test set.

Produces, under data/processed/<case_name>/:
    test.npz          -- shared sparse-grid test set: x, w (weights), y_u, y_p
    train_m{m}_seed{s}.npz  -- one file per (m, seed) combo: x, y_u, y_p
    mass_matrix_u.npz / mass_matrix_p.npz -- FEM mass matrices for u, p
    metadata.json       -- mesh/DOF/sparse-grid/Newton-convergence provenance

Solves within one batch (all test points, or all max(TRAIN_SIZES) draws for
one seed) are independent of each other, so they run across a process pool
instead of serially -- each worker builds its own mesh once (post-fork, so
no FEniCS/PETSc state is shared across the fork boundary) and reuses it for
every solve it's assigned. Thread-count env vars are set before any
numpy/dolfin import so PETSc's linear solver doesn't oversubscribe cores
underneath the process-level parallelism.

Resume: a seed is skipped if all of its train_m{m}_seed{s}.npz files
already exist, and test.npz/mass-matrix files are skipped if already
present, so re-running this script after an interruption only redoes the
work that wasn't finished.

Newton non-convergence handling: failed solves are retried once with a
damped relaxation factor; if still non-convergent, the sample is logged
and excluded (not silently dropped -- see the "excluded_samples" metadata
field), per the documented Phase 4/7 retry policy.

Usage (inside the WSL 'fenics2019' conda env):
    python scripts/wsl/generate_nsb_case.py \
        --case-name nsb_affine_d4 --coefficient affine --dimension 4 --workers 28
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ol_reproduction.data.dataset_io import save_npz_dataset  # noqa: E402
from ol_reproduction.data.sampling import sample_uniform_parameters  # noqa: E402
from ol_reproduction.pde.diffusion.fenics_mixed_solver import load_original_mesh  # noqa: E402
from ol_reproduction.pde.mass_matrix import (  # noqa: E402
    assemble_mass_matrix_csr,
    save_mass_matrix_npz,
)
from ol_reproduction.pde.navier_stokes_brinkman.fenics_solver import (  # noqa: E402
    build_nsb_function_space,
    solve_nsb_mixed_fenics,
)

TRAIN_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
SEEDS = list(range(12))
FE_DEGREE = 1  # matches the authors' --FE_degree default; same mesh as diffusion

_worker_mesh = None  # built once per worker process, in _pool_worker_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--coefficient", required=True, choices=["affine", "log"])
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--sparse-grid-level", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 4),
        help="Process-pool size for parallel solves. Defaults to cpu_count - 4.",
    )
    return parser.parse_args()


def _pool_worker_init() -> None:
    """Runs once per worker process: build this worker's own mesh so no
    dolfin/PETSc state is shared across the fork boundary."""
    global _worker_mesh
    _worker_mesh = load_original_mesh()


def _pool_solve_one(task: tuple[str, np.ndarray]):
    """Solve one NSB sample (with retry) in a worker process."""
    coefficient_name, parameter_vector = task

    result = solve_nsb_mixed_fenics(
        mesh=_worker_mesh, coefficient_name=coefficient_name, parameters=parameter_vector
    )
    if result.converged:
        return result.u_dofs.astype(np.float32), result.p_dofs.astype(np.float32), 1, False

    result_retry = solve_nsb_mixed_fenics(
        mesh=_worker_mesh,
        coefficient_name=coefficient_name,
        parameters=parameter_vector,
        newton_relaxation=0.5,
        newton_max_iterations=80,
    )
    if result_retry.converged:
        return result_retry.u_dofs.astype(np.float32), result_retry.p_dofs.astype(np.float32), 2, False

    return result_retry.u_dofs.astype(np.float32), result_retry.p_dofs.astype(np.float32), 2, True


def _seed_complete(output_dir: Path, seed: int) -> bool:
    return all((output_dir / f"train_m{m}_seed{seed}.npz").exists() for m in TRAIN_SIZES)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "data/processed" / args.case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading the authors' original mesh (data/original_mesh/poisson.xml)...")
    mesh = load_original_mesh()

    function_space, _ = build_nsb_function_space(mesh, fe_degree=FE_DEGREE)
    u_space = function_space.sub(0).collapse()
    if not (output_dir / "mass_matrix_u.npz").exists():
        print("Assembling u mass matrix...")
        save_mass_matrix_npz(output_dir / "mass_matrix_u.npz", assemble_mass_matrix_csr(u_space))

    import dolfin as df

    p_space = df.FunctionSpace(mesh, "DG", 0)
    if not (output_dir / "mass_matrix_p.npz").exists():
        print("Assembling p mass matrix...")
        save_mass_matrix_npz(output_dir / "mass_matrix_p.npz", assemble_mass_matrix_csr(p_space))

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
            f"'sparsegrid' conda env): python scripts/wsl/generate_sparse_grid_points.py "
            f"--dimension {args.dimension} --level {level}"
        )
    quadrature = load_precomputed_quadrature(sparse_grid_path)
    validate_quadrature(quadrature)
    print(f"Sparse grid: dimension={args.dimension} level={level} m_test={quadrature.points.shape[0]}")
    print(f"Worker pool size: {args.workers}")

    excluded_log: list[dict] = []

    def solve_batch_parallel(parameters: np.ndarray, label: str, executor: ProcessPoolExecutor):
        start = time.perf_counter()
        futures = {
            executor.submit(_pool_solve_one, (args.coefficient, parameter_vector)): i
            for i, parameter_vector in enumerate(parameters)
        }
        u_solutions: list = [None] * len(parameters)
        p_solutions: list = [None] * len(parameters)
        done_count = 0
        for future in as_completed(futures):
            i = futures[future]
            u_dofs, p_dofs, _attempts, excluded = future.result()
            if excluded:
                excluded_log.append({"label": label, "index": i, "parameters": parameters[i].tolist()})
            u_solutions[i] = u_dofs
            p_solutions[i] = p_dofs
            done_count += 1
            if done_count % 100 == 0 or done_count == len(parameters):
                elapsed = time.perf_counter() - start
                print(f"  [{label}] {done_count}/{len(parameters)} solves, {elapsed:.0f}s elapsed")
        return np.stack(u_solutions, axis=0), np.stack(p_solutions, axis=0)

    train_elapsed = 0.0
    total_train_solves = len(SEEDS) * max(TRAIN_SIZES)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_pool_worker_init) as executor:
        if (output_dir / "test.npz").exists():
            print("Skipping test set: test.npz already exists.")
        else:
            print(f"Solving {quadrature.points.shape[0]} test points...")
            test_u, test_p = solve_batch_parallel(quadrature.points, "test", executor)
            save_npz_dataset(
                path=output_dir / "test.npz",
                arrays={
                    "x": quadrature.points.astype(np.float32),
                    "w": quadrature.weights.astype(np.float64),
                    "y_u": test_u,
                    "y_p": test_p,
                },
            )

        # Each seed solves only max(TRAIN_SIZES) points once; smaller m-value
        # files are prefix slices of that one solve, not separately re-solved.
        print(
            f"Solving up to {total_train_solves} training points across {len(SEEDS)} seeds "
            f"({max(TRAIN_SIZES)} solves/seed, sliced into {len(TRAIN_SIZES)} m-value files each)..."
        )
        train_start = time.perf_counter()
        for seed in SEEDS:
            if _seed_complete(output_dir, seed):
                print(f"  [train_seed{seed}] already complete, skipping.")
                continue

            parameters = sample_uniform_parameters(
                num_samples=max(TRAIN_SIZES), dimension=args.dimension, seed=seed, dtype=np.float32
            )
            u_solutions, p_solutions = solve_batch_parallel(parameters, f"train_seed{seed}", executor)

            for m in TRAIN_SIZES:
                save_npz_dataset(
                    path=output_dir / f"train_m{m}_seed{seed}.npz",
                    arrays={"x": parameters[:m], "y_u": u_solutions[:m], "y_p": p_solutions[:m]},
                )
        train_elapsed = time.perf_counter() - train_start

    metadata = {
        "case_name": args.case_name,
        "coefficient": args.coefficient,
        "dimension": args.dimension,
        "mesh": {
            "method": "original_authors_mesh",
            "source": "data/original_mesh/poisson.xml",
            "fe_degree": FE_DEGREE,
            "num_cells": mesh.num_cells(),
            "num_dofs_u": u_space.dim(),
            "num_dofs_p": p_space.dim(),
        },
        "sparse_grid": {
            "toolkit": "tasmanian",
            "rule": "clenshaw-curtis",
            "level": level,
            "m_test": int(quadrature.points.shape[0]),
        },
        "train_sizes": TRAIN_SIZES,
        "seeds": SEEDS,
        "newton": {
            "excluded_sample_count": len(excluded_log),
            "excluded_samples": excluded_log,
        },
        "timing": {
            "train_solve_seconds": train_elapsed,
            "total_solves": total_train_solves + int(quadrature.points.shape[0]),
            "workers": args.workers,
        },
        "note": (
            "Generated with the nonlinear mixed FEM NSB solver "
            "(pde/navier_stokes_brinkman/fenics_solver.py), paper eq. "
            "B.14-B.15, AFW elements (BDM2/DG1vec/DG1/DG2vec3) on the "
            "paper authors' own mesh (data/original_mesh/poisson.xml), "
            "matching PDE_DATA/CODE_NSB exactly."
        ),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Done. Output: {output_dir}. Excluded (non-converged) samples: {len(excluded_log)}")


if __name__ == "__main__":
    main()
