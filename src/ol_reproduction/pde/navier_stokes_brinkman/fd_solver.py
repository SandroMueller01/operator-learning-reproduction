"""Simplified finite-difference solver for a Brinkman-type flow problem.

This module implements a stabilized linear Brinkman approximation used for the
practical reproduction pipeline. It is not an exact implementation of the mixed
finite element Navier--Stokes--Brinkman solver used in the target paper.

The solved model is a simplified stationary Brinkman system,

    -nu(x) Delta u + alpha u + grad p = f,
     div u - epsilon p = 0,

with homogeneous velocity boundary conditions. The pressure stabilization term
``epsilon p`` removes the pressure nullspace and keeps the dense linear system
well-conditioned for small grid sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NsbGrid:
    """Structured grid definition for the unit square."""

    nx: int
    ny: int

    def __post_init__(self) -> None:
        """Validate grid dimensions."""
        if self.nx < 4:
            raise ValueError("nx must be at least 4.")

        if self.ny < 4:
            raise ValueError("ny must be at least 4.")

    @property
    def hx(self) -> float:
        """Grid spacing in x-direction."""
        return 1.0 / (self.nx - 1)

    @property
    def hy(self) -> float:
        """Grid spacing in y-direction."""
        return 1.0 / (self.ny - 1)


@dataclass(frozen=True)
class NsbForcing:
    """Constant forcing for the simplified Brinkman system."""

    force_x: float = 1.0
    force_y: float = 0.0


@dataclass(frozen=True)
class NsbSolverParameters:
    """Numerical parameters for the simplified NSB solver."""

    brinkman_alpha: float = 1.0
    pressure_stabilization: float = 1.0e-6


def create_unit_square_grid(grid: NsbGrid) -> tuple[np.ndarray, np.ndarray]:
    """Create a structured grid on the unit square.

    Parameters
    ----------
    grid:
        Grid configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Coordinate arrays ``z1`` and ``z2`` with shape ``(ny, nx)``.
    """
    x_values = np.linspace(0.0, 1.0, grid.nx)
    y_values = np.linspace(0.0, 1.0, grid.ny)

    z1, z2 = np.meshgrid(x_values, y_values, indexing="xy")
    return z1, z2


def solve_nsb_fd(
    viscosity: np.ndarray,
    forcing: NsbForcing | None = None,
    solver_parameters: NsbSolverParameters | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a simplified finite-difference Brinkman system.

    Parameters
    ----------
    viscosity:
        Positive viscosity field with shape ``(ny, nx)``.
    forcing:
        Constant forcing vector.
    solver_parameters:
        Stabilization and Brinkman parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Velocity field with shape ``(ny, nx, 2)`` and pressure field with shape
        ``(ny, nx)``.
    """
    _validate_viscosity(viscosity)

    if forcing is None:
        forcing = NsbForcing()

    if solver_parameters is None:
        solver_parameters = NsbSolverParameters()

    ny, nx = viscosity.shape
    grid = NsbGrid(nx=nx, ny=ny)

    num_interior = (nx - 2) * (ny - 2)
    num_unknowns = 3 * num_interior

    matrix = np.zeros((num_unknowns, num_unknowns), dtype=np.float64)
    rhs = np.zeros(num_unknowns, dtype=np.float64)

    for row in range(1, ny - 1):
        for col in range(1, nx - 1):
            point_index = _interior_index(row=row, col=col, nx=nx)

            _add_momentum_x_equation(
                matrix=matrix,
                rhs=rhs,
                viscosity=viscosity,
                grid=grid,
                row=row,
                col=col,
                point_index=point_index,
                forcing=forcing,
                solver_parameters=solver_parameters,
            )

            _add_momentum_y_equation(
                matrix=matrix,
                rhs=rhs,
                viscosity=viscosity,
                grid=grid,
                row=row,
                col=col,
                point_index=point_index,
                forcing=forcing,
                solver_parameters=solver_parameters,
            )

            _add_continuity_equation(
                matrix=matrix,
                grid=grid,
                row=row,
                col=col,
                point_index=point_index,
                solver_parameters=solver_parameters,
            )

    solution_vector = np.linalg.solve(matrix, rhs)

    velocity, pressure = _unpack_solution(
        solution_vector=solution_vector,
        nx=nx,
        ny=ny,
    )

    pressure = _zero_mean_pressure(pressure)

    return velocity, pressure


def flatten_velocity(velocity: np.ndarray) -> np.ndarray:
    """Flatten velocity field.

    Parameters
    ----------
    velocity:
        Velocity array with shape ``(ny, nx, 2)``.

    Returns
    -------
    np.ndarray
        Flattened velocity vector.
    """
    if velocity.ndim != 3 or velocity.shape[-1] != 2:
        raise ValueError(
            "velocity must have shape (ny, nx, 2), "
            f"got {velocity.shape}."
        )

    return np.asarray(velocity, dtype=np.float32).reshape(-1)


def flatten_pressure(pressure: np.ndarray) -> np.ndarray:
    """Flatten pressure field.

    Parameters
    ----------
    pressure:
        Pressure array with shape ``(ny, nx)``.

    Returns
    -------
    np.ndarray
        Flattened pressure vector.
    """
    if pressure.ndim != 2:
        raise ValueError(
            "pressure must have shape (ny, nx), "
            f"got {pressure.shape}."
        )

    return np.asarray(pressure, dtype=np.float32).reshape(-1)


def _validate_viscosity(viscosity: np.ndarray) -> None:
    """Validate viscosity field."""
    if viscosity.ndim != 2:
        raise ValueError(
            "viscosity must have shape (ny, nx), "
            f"got {viscosity.shape}."
        )

    if viscosity.shape[0] < 4 or viscosity.shape[1] < 4:
        raise ValueError("viscosity grid must be at least 4 by 4.")

    if np.any(viscosity <= 0.0):
        raise ValueError("viscosity must be strictly positive.")


def _interior_index(row: int, col: int, nx: int) -> int:
    """Map an interior grid point to a scalar interior index."""
    return (row - 1) * (nx - 2) + (col - 1)


def _u_index(point_index: int) -> int:
    """Return global index for horizontal velocity component."""
    return point_index


def _v_index(point_index: int, num_interior: int) -> int:
    """Return global index for vertical velocity component."""
    return num_interior + point_index


def _p_index(point_index: int, num_interior: int) -> int:
    """Return global index for pressure."""
    return 2 * num_interior + point_index


def _add_momentum_x_equation(
    matrix: np.ndarray,
    rhs: np.ndarray,
    viscosity: np.ndarray,
    grid: NsbGrid,
    row: int,
    col: int,
    point_index: int,
    forcing: NsbForcing,
    solver_parameters: NsbSolverParameters,
) -> None:
    """Add horizontal momentum equation."""
    num_interior = (grid.nx - 2) * (grid.ny - 2)
    equation_index = _u_index(point_index)
    center_index = _u_index(point_index)

    viscosity_value = viscosity[row, col]
    diagonal = (
        2.0 * viscosity_value / grid.hx**2
        + 2.0 * viscosity_value / grid.hy**2
        + solver_parameters.brinkman_alpha
    )

    matrix[equation_index, center_index] = diagonal
    rhs[equation_index] = forcing.force_x

    _add_laplacian_neighbors(
        matrix=matrix,
        equation_index=equation_index,
        variable="u",
        viscosity_value=viscosity_value,
        grid=grid,
        row=row,
        col=col,
    )

    _add_pressure_gradient_x(
        matrix=matrix,
        equation_index=equation_index,
        num_interior=num_interior,
        grid=grid,
        row=row,
        col=col,
    )


def _add_momentum_y_equation(
    matrix: np.ndarray,
    rhs: np.ndarray,
    viscosity: np.ndarray,
    grid: NsbGrid,
    row: int,
    col: int,
    point_index: int,
    forcing: NsbForcing,
    solver_parameters: NsbSolverParameters,
) -> None:
    """Add vertical momentum equation."""
    num_interior = (grid.nx - 2) * (grid.ny - 2)
    equation_index = _v_index(point_index, num_interior)
    center_index = _v_index(point_index, num_interior)

    viscosity_value = viscosity[row, col]
    diagonal = (
        2.0 * viscosity_value / grid.hx**2
        + 2.0 * viscosity_value / grid.hy**2
        + solver_parameters.brinkman_alpha
    )

    matrix[equation_index, center_index] = diagonal
    rhs[equation_index] = forcing.force_y

    _add_laplacian_neighbors(
        matrix=matrix,
        equation_index=equation_index,
        variable="v",
        viscosity_value=viscosity_value,
        grid=grid,
        row=row,
        col=col,
    )

    _add_pressure_gradient_y(
        matrix=matrix,
        equation_index=equation_index,
        num_interior=num_interior,
        grid=grid,
        row=row,
        col=col,
    )


def _add_laplacian_neighbors(
    matrix: np.ndarray,
    equation_index: int,
    variable: str,
    viscosity_value: float,
    grid: NsbGrid,
    row: int,
    col: int,
) -> None:
    """Add Laplacian neighbor entries for velocity components."""
    num_interior = (grid.nx - 2) * (grid.ny - 2)

    neighbors = [
        (row, col + 1, -viscosity_value / grid.hx**2),
        (row, col - 1, -viscosity_value / grid.hx**2),
        (row + 1, col, -viscosity_value / grid.hy**2),
        (row - 1, col, -viscosity_value / grid.hy**2),
    ]

    for neighbor_row, neighbor_col, weight in neighbors:
        if _is_boundary(
            row=neighbor_row,
            col=neighbor_col,
            nx=grid.nx,
            ny=grid.ny,
        ):
            continue

        neighbor_point_index = _interior_index(
            row=neighbor_row,
            col=neighbor_col,
            nx=grid.nx,
        )

        if variable == "u":
            neighbor_index = _u_index(neighbor_point_index)
        elif variable == "v":
            neighbor_index = _v_index(neighbor_point_index, num_interior)
        else:
            raise ValueError(f"Unsupported velocity variable: {variable}")

        matrix[equation_index, neighbor_index] = weight


def _add_pressure_gradient_x(
    matrix: np.ndarray,
    equation_index: int,
    num_interior: int,
    grid: NsbGrid,
    row: int,
    col: int,
) -> None:
    """Add central-difference pressure gradient in x-direction."""
    east = (row, col + 1, 1.0 / (2.0 * grid.hx))
    west = (row, col - 1, -1.0 / (2.0 * grid.hx))

    for neighbor_row, neighbor_col, weight in [east, west]:
        if _is_boundary(neighbor_row, neighbor_col, grid.nx, grid.ny):
            continue

        neighbor_point_index = _interior_index(
            row=neighbor_row,
            col=neighbor_col,
            nx=grid.nx,
        )
        pressure_index = _p_index(neighbor_point_index, num_interior)
        matrix[equation_index, pressure_index] += weight


def _add_pressure_gradient_y(
    matrix: np.ndarray,
    equation_index: int,
    num_interior: int,
    grid: NsbGrid,
    row: int,
    col: int,
) -> None:
    """Add central-difference pressure gradient in y-direction."""
    north = (row + 1, col, 1.0 / (2.0 * grid.hy))
    south = (row - 1, col, -1.0 / (2.0 * grid.hy))

    for neighbor_row, neighbor_col, weight in [north, south]:
        if _is_boundary(neighbor_row, neighbor_col, grid.nx, grid.ny):
            continue

        neighbor_point_index = _interior_index(
            row=neighbor_row,
            col=neighbor_col,
            nx=grid.nx,
        )
        pressure_index = _p_index(neighbor_point_index, num_interior)
        matrix[equation_index, pressure_index] += weight


def _add_continuity_equation(
    matrix: np.ndarray,
    grid: NsbGrid,
    row: int,
    col: int,
    point_index: int,
    solver_parameters: NsbSolverParameters,
) -> None:
    """Add stabilized continuity equation."""
    num_interior = (grid.nx - 2) * (grid.ny - 2)
    equation_index = _p_index(point_index, num_interior)

    _add_velocity_derivative_x(
        matrix=matrix,
        equation_index=equation_index,
        grid=grid,
        row=row,
        col=col,
    )
    _add_velocity_derivative_y(
        matrix=matrix,
        equation_index=equation_index,
        grid=grid,
        row=row,
        col=col,
    )

    pressure_index = _p_index(point_index, num_interior)
    matrix[equation_index, pressure_index] = (
        -solver_parameters.pressure_stabilization
    )


def _add_velocity_derivative_x(
    matrix: np.ndarray,
    equation_index: int,
    grid: NsbGrid,
    row: int,
    col: int,
) -> None:
    """Add du/dx term to continuity equation."""
    east = (row, col + 1, 1.0 / (2.0 * grid.hx))
    west = (row, col - 1, -1.0 / (2.0 * grid.hx))

    for neighbor_row, neighbor_col, weight in [east, west]:
        if _is_boundary(neighbor_row, neighbor_col, grid.nx, grid.ny):
            continue

        neighbor_point_index = _interior_index(
            row=neighbor_row,
            col=neighbor_col,
            nx=grid.nx,
        )
        matrix[equation_index, _u_index(neighbor_point_index)] += weight


def _add_velocity_derivative_y(
    matrix: np.ndarray,
    equation_index: int,
    grid: NsbGrid,
    row: int,
    col: int,
) -> None:
    """Add dv/dy term to continuity equation."""
    num_interior = (grid.nx - 2) * (grid.ny - 2)

    north = (row + 1, col, 1.0 / (2.0 * grid.hy))
    south = (row - 1, col, -1.0 / (2.0 * grid.hy))

    for neighbor_row, neighbor_col, weight in [north, south]:
        if _is_boundary(neighbor_row, neighbor_col, grid.nx, grid.ny):
            continue

        neighbor_point_index = _interior_index(
            row=neighbor_row,
            col=neighbor_col,
            nx=grid.nx,
        )
        velocity_index = _v_index(neighbor_point_index, num_interior)
        matrix[equation_index, velocity_index] += weight


def _is_boundary(row: int, col: int, nx: int, ny: int) -> bool:
    """Check whether a grid point is on the boundary."""
    return row == 0 or row == ny - 1 or col == 0 or col == nx - 1


def _unpack_solution(
    solution_vector: np.ndarray,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack linear-system solution into velocity and pressure fields."""
    num_interior = (nx - 2) * (ny - 2)

    velocity = np.zeros((ny, nx, 2), dtype=np.float64)
    pressure = np.zeros((ny, nx), dtype=np.float64)

    u_values = solution_vector[:num_interior]
    v_values = solution_vector[num_interior : 2 * num_interior]
    p_values = solution_vector[2 * num_interior :]

    for row in range(1, ny - 1):
        for col in range(1, nx - 1):
            point_index = _interior_index(row=row, col=col, nx=nx)
            velocity[row, col, 0] = u_values[point_index]
            velocity[row, col, 1] = v_values[point_index]
            pressure[row, col] = p_values[point_index]

    return velocity, pressure


def _zero_mean_pressure(pressure: np.ndarray) -> np.ndarray:
    """Normalize pressure to have zero mean over the full grid."""
    pressure = pressure.copy()
    pressure -= np.mean(pressure)
    return pressure