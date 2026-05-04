"""Dataset generation using FEniCS-based solvers.

At the moment, this module provides a usable FEniCS path for the diffusion
benchmark and explicit placeholders for NSB and Boussinesq. The NSB and
Boussinesq paper solvers require full nonlinear mixed FEM implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ol_reproduction.data.dataset_io import save_metadata, save_npz_dataset
from ol_reproduction.data.sampling import sample_uniform_parameters
from ol_reproduction.pde.diffusion.fenics_solver import (
    FenicsDiffusionConfig,
    solve_diffusion_fenics,
)
from ol_reproduction.pde.navier_stokes_brinkman.fenics_solver import (
    FenicsNsbConfig,
    solve_nsb_fenics,
)
from ol_reproduction.pde.boussinesq.fenics_solver import (
    FenicsBoussinesqConfig,
    solve_boussinesq_fenics,
)


ConfigDict = dict[str, Any]


def generate_fenics_dataset_from_config(config: ConfigDict) -> None:
    """Generate a dataset using the configured FEniCS solver.

    Parameters
    ----------
    config:
        PDE configuration dictionary.

    Raises
    ------
    ValueError
        If the configured problem is unsupported.
    """
    problem = str(config["experiment"]["problem"])

    if problem == "diffusion":
        generate_fenics_diffusion_dataset_from_config(config)
        return

    if problem == "navier_stokes_brinkman":
        generate_fenics_nsb_dataset_from_config(config)
        return

    if problem == "boussinesq":
        generate_fenics_boussinesq_dataset_from_config(config)
        return

    raise ValueError(f"Unsupported problem for FEniCS generation: {problem}")


def generate_fenics_diffusion_dataset_from_config(config: ConfigDict) -> None:
    """Generate diffusion train/test datasets using FEniCS."""
    experiment_config = config["experiment"]
    coefficient_config = config["coefficient"]
    data_config = config["data"]
    path_config = config["paths"]
    pde_config = config["pde"]

    dimension = int(experiment_config["dimension"])
    train_sizes = [int(size) for size in data_config["train_sizes"]]
    test_size = int(data_config["test_size"])
    seeds = [int(seed) for seed in data_config["seeds"]]
    dtype = str(data_config.get("dtype", "float32"))

    coefficient_name = str(coefficient_config["name"])
    output_dir = Path(path_config["output_dir"] + "_fenics")

    fenics_config = FenicsDiffusionConfig(
        mesh_resolution=int(config.get("fenics", {}).get("mesh_resolution", 32)),
        forcing=float(pde_config.get("forcing", 10.0)),
        bottom_value=float(
            pde_config["boundary_conditions"].get("bottom", 0.5)
        ),
        coefficient_name=coefficient_name,
        affine_base_value=float(coefficient_config.get("base_value", 2.62)),
        log_base_shift=float(coefficient_config.get("base_shift", 1.0)),
        dtype=dtype,
    )

    for seed in seeds:
        test_dataset = _generate_single_fenics_diffusion_dataset(
            num_samples=test_size,
            dimension=dimension,
            seed=seed + 10_000,
            solver_config=fenics_config,
            dtype=dtype,
        )
        save_npz_dataset(
            path=output_dir / f"test_seed{seed}.npz",
            arrays=test_dataset,
        )

        for train_size in train_sizes:
            train_dataset = _generate_single_fenics_diffusion_dataset(
                num_samples=train_size,
                dimension=dimension,
                seed=seed,
                solver_config=fenics_config,
                dtype=dtype,
            )
            save_npz_dataset(
                path=output_dir / f"train_m{train_size}_seed{seed}.npz",
                arrays=train_dataset,
            )

    save_metadata(
        path=output_dir / "metadata.json",
        metadata={
            "experiment": config["experiment"],
            "coefficient": config["coefficient"],
            "data": config["data"],
            "solver": "fenics_diffusion",
            "note": (
                "FEniCS diffusion dataset. Output dimension is determined by "
                "the FEniCS function space degrees of freedom."
            ),
        },
    )


def generate_fenics_nsb_dataset_from_config(config: ConfigDict) -> None:
    """Generate NSB dataset using FEniCS.

    This currently raises ``NotImplementedError`` through the solver.
    """
    dimension = int(config["experiment"]["dimension"])
    parameters = np.zeros(dimension, dtype=np.float64)

    solve_nsb_fenics(
        parameters=parameters,
        config=FenicsNsbConfig(),
    )


def generate_fenics_boussinesq_dataset_from_config(config: ConfigDict) -> None:
    """Generate Boussinesq dataset using FEniCS.

    This currently raises ``NotImplementedError`` through the solver.
    """
    dimension = int(config["experiment"]["dimension"])
    parameters = np.zeros(dimension, dtype=np.float64)

    solve_boussinesq_fenics(
        parameters=parameters,
        config=FenicsBoussinesqConfig(),
    )


def _generate_single_fenics_diffusion_dataset(
    num_samples: int,
    dimension: int,
    seed: int,
    solver_config: FenicsDiffusionConfig,
    dtype: str,
) -> dict[str, np.ndarray]:
    """Generate one FEniCS diffusion dataset split."""
    numpy_dtype = _resolve_numpy_dtype(dtype)

    parameters = sample_uniform_parameters(
        num_samples=num_samples,
        dimension=dimension,
        seed=seed,
        dtype=numpy_dtype,
    )

    solutions = []

    for parameter_vector in parameters:
        solution = solve_diffusion_fenics(
            parameters=parameter_vector,
            config=solver_config,
        )
        solutions.append(solution)

    solution_array = np.stack(solutions, axis=0).astype(numpy_dtype)

    return {
        "x": parameters.astype(numpy_dtype),
        "y_u": solution_array,
    }


def _resolve_numpy_dtype(dtype_name: str) -> np.dtype:
    """Resolve dtype name to NumPy dtype."""
    if dtype_name == "float32":
        return np.float32

    if dtype_name == "float64":
        return np.float64

    raise ValueError(f"Unsupported dtype: {dtype_name}")