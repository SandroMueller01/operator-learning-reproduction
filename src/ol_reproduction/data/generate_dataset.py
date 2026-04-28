"""Dataset generation for PDE reproduction experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ol_reproduction.coefficients.affine import affine_coefficient
from ol_reproduction.coefficients.log_transformed import (
    log_transformed_coefficient,
)
from ol_reproduction.data.dataset_io import save_metadata, save_npz_dataset
from ol_reproduction.data.sampling import sample_uniform_parameters
from ol_reproduction.pde.diffusion.fd_solver import (
    DiffusionBoundaryConditions,
    DiffusionGrid,
    create_unit_square_grid as create_diffusion_grid,
    flatten_solution,
    solve_diffusion_fd,
)
from ol_reproduction.pde.navier_stokes_brinkman.fd_solver import (
    NsbForcing,
    NsbGrid,
    NsbSolverParameters,
    create_unit_square_grid as create_nsb_grid,
    flatten_pressure,
    flatten_velocity,
    solve_nsb_fd,
)

from ol_reproduction.pde.boussinesq.fd_solver import (
    BoussinesqSolverParameters,
    flatten_boussinesq_pressure,
    flatten_boussinesq_velocity,
    flatten_temperature,
    solve_boussinesq_fd,
)

ConfigDict = dict[str, Any]


def generate_diffusion_dataset_from_config(config: ConfigDict) -> None:
    """Generate train and test datasets for the diffusion experiment.

    Parameters
    ----------
    config:
        PDE configuration dictionary loaded from YAML.
    """
    experiment_config = config["experiment"]
    grid_config = config["grid"]
    pde_config = config["pde"]
    coefficient_config = config["coefficient"]
    data_config = config["data"]
    path_config = config["paths"]

    dimension = int(experiment_config["dimension"])
    train_sizes = [int(size) for size in data_config["train_sizes"]]
    test_size = int(data_config["test_size"])
    seeds = [int(seed) for seed in data_config["seeds"]]
    dtype = _resolve_numpy_dtype(data_config.get("dtype", "float32"))

    coefficient_name = str(coefficient_config["name"])
    coefficient_base_value = float(coefficient_config.get("base_value", 2.62))

    output_dir = Path(path_config["output_dir"])
    grid = DiffusionGrid(
        nx=int(grid_config["nx"]),
        ny=int(grid_config["ny"]),
    )
    boundary_conditions = _build_diffusion_boundary_conditions(pde_config)
    forcing = float(pde_config["forcing"])

    z1, _ = create_diffusion_grid(grid)

    for seed in seeds:
        test_dataset = _generate_single_diffusion_dataset(
            num_samples=test_size,
            dimension=dimension,
            seed=seed + 10_000,
            z1=z1,
            forcing=forcing,
            boundary_conditions=boundary_conditions,
            dtype=dtype,
            coefficient_name=coefficient_name,
            coefficient_base_value=coefficient_base_value,
        )
        save_npz_dataset(
            path=output_dir / f"test_seed{seed}.npz",
            arrays=test_dataset,
        )

        for train_size in train_sizes:
            train_dataset = _generate_single_diffusion_dataset(
                num_samples=train_size,
                dimension=dimension,
                seed=seed,
                z1=z1,
                forcing=forcing,
                boundary_conditions=boundary_conditions,
                dtype=dtype,
                coefficient_name=coefficient_name,
                coefficient_base_value=coefficient_base_value,
            )
            save_npz_dataset(
                path=output_dir / f"train_m{train_size}_seed{seed}.npz",
                arrays=train_dataset,
            )

    metadata = _build_metadata(
        config=config,
        grid_shape=(grid.ny, grid.nx),
        output_dimensions={
            "y_u": grid.nx * grid.ny,
        },
        note="Diffusion dataset generated with finite differences.",
    )
    save_metadata(
        path=output_dir / "metadata.json",
        metadata=metadata,
    )


def generate_nsb_dataset_from_config(config: ConfigDict) -> None:
    """Generate train and test datasets for the simplified NSB experiment.

    Parameters
    ----------
    config:
        PDE configuration dictionary loaded from YAML.
    """
    experiment_config = config["experiment"]
    grid_config = config["grid"]
    coefficient_config = config["coefficient"]
    data_config = config["data"]
    path_config = config["paths"]

    dimension = int(experiment_config["dimension"])
    train_sizes = [int(size) for size in data_config["train_sizes"]]
    test_size = int(data_config["test_size"])
    seeds = [int(seed) for seed in data_config["seeds"]]
    dtype = _resolve_numpy_dtype(data_config.get("dtype", "float32"))

    coefficient_name = str(coefficient_config["name"])
    coefficient_base_value = float(coefficient_config.get("base_value", 2.62))

    output_dir = Path(path_config["output_dir"])
    grid = NsbGrid(
        nx=int(grid_config["nx"]),
        ny=int(grid_config["ny"]),
    )

    z1, _ = create_nsb_grid(grid)

    for seed in seeds:
        test_dataset = _generate_single_nsb_dataset(
            num_samples=test_size,
            dimension=dimension,
            seed=seed + 10_000,
            z1=z1,
            dtype=dtype,
            coefficient_name=coefficient_name,
            coefficient_base_value=coefficient_base_value,
        )
        save_npz_dataset(
            path=output_dir / f"test_seed{seed}.npz",
            arrays=test_dataset,
        )

        for train_size in train_sizes:
            train_dataset = _generate_single_nsb_dataset(
                num_samples=train_size,
                dimension=dimension,
                seed=seed,
                z1=z1,
                dtype=dtype,
                coefficient_name=coefficient_name,
                coefficient_base_value=coefficient_base_value,
            )
            save_npz_dataset(
                path=output_dir / f"train_m{train_size}_seed{seed}.npz",
                arrays=train_dataset,
            )

    metadata = _build_metadata(
        config=config,
        grid_shape=(grid.ny, grid.nx),
        output_dimensions={
            "y_u": grid.nx * grid.ny * 2,
            "y_p": grid.nx * grid.ny,
        },
        note=(
            "Simplified stabilized finite-difference Brinkman dataset. "
            "This is not the exact mixed FEM NSB solver from the target paper."
        ),
    )
    save_metadata(
        path=output_dir / "metadata.json",
        metadata=metadata,
    )

def generate_boussinesq_dataset_from_config(config: ConfigDict) -> None:
    """Generate train and test datasets for the simplified Boussinesq problem.

    Parameters
    ----------
    config:
        PDE configuration dictionary loaded from YAML.
    """
    experiment_config = config["experiment"]
    grid_config = config["grid"]
    coefficient_config = config["coefficient"]
    data_config = config["data"]
    path_config = config["paths"]

    dimension = int(experiment_config["dimension"])
    train_sizes = [int(size) for size in data_config["train_sizes"]]
    test_size = int(data_config["test_size"])
    seeds = [int(seed) for seed in data_config["seeds"]]
    dtype = _resolve_numpy_dtype(data_config.get("dtype", "float32"))

    coefficient_name = str(coefficient_config["name"])
    coefficient_base_value = float(coefficient_config.get("base_value", 2.62))

    output_dir = Path(path_config["output_dir"])

    nx = int(grid_config["nx"])
    ny = int(grid_config["ny"])

    z1 = _create_2d_z1_grid(nx=nx, ny=ny)

    for seed in seeds:
        test_dataset = _generate_single_boussinesq_dataset(
            num_samples=test_size,
            dimension=dimension,
            seed=seed + 10_000,
            z1=z1,
            dtype=dtype,
            coefficient_name=coefficient_name,
            coefficient_base_value=coefficient_base_value,
        )
        save_npz_dataset(
            path=output_dir / f"test_seed{seed}.npz",
            arrays=test_dataset,
        )

        for train_size in train_sizes:
            train_dataset = _generate_single_boussinesq_dataset(
                num_samples=train_size,
                dimension=dimension,
                seed=seed,
                z1=z1,
                dtype=dtype,
                coefficient_name=coefficient_name,
                coefficient_base_value=coefficient_base_value,
            )
            save_npz_dataset(
                path=output_dir / f"train_m{train_size}_seed{seed}.npz",
                arrays=train_dataset,
            )

    metadata = _build_metadata(
        config=config,
        grid_shape=(ny, nx),
        output_dimensions={
            "y_u": nx * ny * 2,
            "y_phi": nx * ny,
            "y_p": nx * ny,
        },
        note=(
            "Simplified finite-difference Boussinesq-style dataset. "
            "This is not the exact mixed FEM Boussinesq solver from the "
            "target paper. If the config specifies a 3D unit cube, this "
            "implementation uses a 2D slice with nx by ny grid points."
        ),
    )
    save_metadata(
        path=output_dir / "metadata.json",
        metadata=metadata,
    )

def _generate_single_diffusion_dataset(
    num_samples: int,
    dimension: int,
    seed: int,
    z1: np.ndarray,
    forcing: float,
    boundary_conditions: DiffusionBoundaryConditions,
    dtype: np.dtype,
    coefficient_name: str,
    coefficient_base_value: float,
) -> dict[str, np.ndarray]:
    """Generate one diffusion dataset split."""
    parameters = sample_uniform_parameters(
        num_samples=num_samples,
        dimension=dimension,
        seed=seed,
        dtype=dtype,
    )

    solutions = []

    for parameter_vector in parameters:
        coefficient = _evaluate_coefficient(
            coefficient_name=coefficient_name,
            z1=z1,
            parameters=parameter_vector,
            base_value=coefficient_base_value,
        )
        solution = solve_diffusion_fd(
            coefficient=coefficient,
            forcing=forcing,
            boundary_conditions=boundary_conditions,
        )
        solutions.append(flatten_solution(solution))

    solution_array = np.stack(solutions, axis=0).astype(dtype)

    return {
        "x": parameters.astype(dtype),
        "y_u": solution_array,
    }


def _generate_single_nsb_dataset(
    num_samples: int,
    dimension: int,
    seed: int,
    z1: np.ndarray,
    dtype: np.dtype,
    coefficient_name: str,
    coefficient_base_value: float,
) -> dict[str, np.ndarray]:
    """Generate one simplified NSB dataset split."""
    parameters = sample_uniform_parameters(
        num_samples=num_samples,
        dimension=dimension,
        seed=seed,
        dtype=dtype,
    )

    velocities = []
    pressures = []

    for parameter_vector in parameters:
        viscosity = _evaluate_coefficient(
            coefficient_name=coefficient_name,
            z1=z1,
            parameters=parameter_vector,
            base_value=coefficient_base_value,
        )
        velocity, pressure = solve_nsb_fd(
            viscosity=viscosity,
            forcing=NsbForcing(),
            solver_parameters=NsbSolverParameters(),
        )
        velocities.append(flatten_velocity(velocity))
        pressures.append(flatten_pressure(pressure))

    velocity_array = np.stack(velocities, axis=0).astype(dtype)
    pressure_array = np.stack(pressures, axis=0).astype(dtype)

    return {
        "x": parameters.astype(dtype),
        "y_u": velocity_array,
        "y_p": pressure_array,
    }

def _generate_single_boussinesq_dataset(
    num_samples: int,
    dimension: int,
    seed: int,
    z1: np.ndarray,
    dtype: np.dtype,
    coefficient_name: str,
    coefficient_base_value: float,
) -> dict[str, np.ndarray]:
    """Generate one simplified Boussinesq dataset split.

    Parameters
    ----------
    num_samples:
        Number of samples.
    dimension:
        Parametric dimension.
    seed:
        Random seed.
    z1:
        First spatial coordinate grid.
    dtype:
        Output dtype.
    coefficient_name:
        Name of the coefficient type.
    coefficient_base_value:
        Positive baseline coefficient value.

    Returns
    -------
    dict[str, np.ndarray]
        Dataset containing ``x``, ``y_u``, ``y_phi`` and ``y_p``.
    """
    parameters = sample_uniform_parameters(
        num_samples=num_samples,
        dimension=dimension,
        seed=seed,
        dtype=dtype,
    )

    velocities = []
    temperatures = []
    pressures = []

    for parameter_vector in parameters:
        viscosity = _evaluate_coefficient(
            coefficient_name=coefficient_name,
            z1=z1,
            parameters=parameter_vector,
            base_value=coefficient_base_value,
        )

        thermal_conductivity = _evaluate_coefficient(
            coefficient_name=coefficient_name,
            z1=z1,
            parameters=parameter_vector[::-1],
            base_value=coefficient_base_value,
        )

        velocity, temperature, pressure = solve_boussinesq_fd(
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            solver_parameters=BoussinesqSolverParameters(),
        )

        velocities.append(flatten_boussinesq_velocity(velocity))
        temperatures.append(flatten_temperature(temperature))
        pressures.append(flatten_boussinesq_pressure(pressure))

    velocity_array = np.stack(velocities, axis=0).astype(dtype)
    temperature_array = np.stack(temperatures, axis=0).astype(dtype)
    pressure_array = np.stack(pressures, axis=0).astype(dtype)

    return {
        "x": parameters.astype(dtype),
        "y_u": velocity_array,
        "y_phi": temperature_array,
        "y_p": pressure_array,
    }

def _evaluate_coefficient(
    coefficient_name: str,
    z1: np.ndarray,
    parameters: np.ndarray,
    base_value: float,
) -> np.ndarray:
    """Evaluate configured coefficient."""
    if coefficient_name == "affine":
        return affine_coefficient(
            z1=z1,
            parameters=parameters,
            base_value=base_value,
        )

    if coefficient_name == "log":
        return log_transformed_coefficient(
            z1=z1,
            parameters=parameters,
            base_value=base_value,
        )

    raise ValueError(f"Unsupported coefficient: {coefficient_name}")


def _build_diffusion_boundary_conditions(
    pde_config: ConfigDict,
) -> DiffusionBoundaryConditions:
    """Build diffusion boundary condition dataclass from config."""
    boundary_config = pde_config["boundary_conditions"]

    return DiffusionBoundaryConditions(
        bottom=float(boundary_config["bottom"]),
        top=float(boundary_config["top"]),
        left=float(boundary_config["left"]),
        right=float(boundary_config["right"]),
    )


def _resolve_numpy_dtype(dtype_name: str) -> np.dtype:
    """Resolve dtype string to NumPy dtype."""
    if dtype_name == "float32":
        return np.float32

    if dtype_name == "float64":
        return np.float64

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _build_metadata(
    config: ConfigDict,
    grid_shape: tuple[int, ...],
    output_dimensions: dict[str, int],
    note: str,
) -> dict[str, Any]:
    """Build dataset metadata."""
    return {
        "experiment": config["experiment"],
        "grid": {
            "shape": list(grid_shape),
        },
        "pde": config["pde"],
        "coefficient": config["coefficient"],
        "data": config["data"],
        "output": {
            "output_dimensions": output_dimensions,
        },
        "note": note,
    }
def _create_2d_z1_grid(nx: int, ny: int) -> np.ndarray:
    """Create the first coordinate grid for a two-dimensional unit square.

    Parameters
    ----------
    nx:
        Number of grid points in x-direction.
    ny:
        Number of grid points in y-direction.

    Returns
    -------
    np.ndarray
        First coordinate grid with shape ``(ny, nx)``.
    """
    x_values = np.linspace(0.0, 1.0, nx)
    y_values = np.linspace(0.0, 1.0, ny)
    z1, _ = np.meshgrid(x_values, y_values, indexing="xy")

    return z1