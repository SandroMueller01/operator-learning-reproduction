"""Finite-difference solver for the parametric diffusion equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiffusionBoundaryConditions:
    """Dirichlet boundary values for the unit-square diffusion problem."""

    bottom: float = 0.5
    top: float = 0.0
    left: float = 0.0
    right: float = 0.0


@dataclass(frozen=True)
class DiffusionGrid:
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


def create_unit_square_grid(grid: DiffusionGrid) -> tuple[np.ndarray, np.ndarray]:
    """Create a structured grid on the unit square.

    Parameters
    ----------
    grid:
        Grid configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Meshgrid arrays ``z1`` and ``z2`` with shape ``(ny, nx)``.
    """
    x_values = np.linspace(0.0, 1.0, grid.nx)
    y_values = np.linspace(0.0, 1.0, grid.ny)

    z1, z2 = np.meshgrid(x_values, y_values, indexing="xy")
    return z1, z2


def apply_dirichlet_boundary(
    solution: np.ndarray,
    boundary_conditions: DiffusionBoundaryConditions,
) -> np.ndarray:
    """Apply Dirichlet boundary conditions to a solution array.

    Parameters
    ----------
    solution:
        Solution array with shape ``(ny, nx)``.
    boundary_conditions:
        Boundary values.

    Returns
    -------
    np.ndarray
        Solution array with boundary values applied.
    """
    solution = solution.copy()

    solution[0, :] = boundary_conditions.bottom
    solution[-1, :] = boundary_conditions.top
    solution[:, 0] = boundary_conditions.left
    solution[:, -1] = boundary_conditions.right

    return solution


def solve_diffusion_fd(
    coefficient: np.ndarray,
    forcing: float,
    boundary_conditions: DiffusionBoundaryConditions,
) -> np.ndarray:
    """Solve the variable-coefficient diffusion equation with finite differences.

    The equation is

    .. math::

        -\\nabla \\cdot (a \\nabla u) = f

    on the unit square with Dirichlet boundary conditions.

    Parameters
    ----------
    coefficient:
        Positive coefficient field with shape ``(ny, nx)``.
    forcing:
        Constant right-hand side ``f``.
    boundary_conditions:
        Dirichlet boundary values.

    Returns
    -------
    np.ndarray
        Solution field with shape ``(ny, nx)``.

    Raises
    ------
    ValueError
        If the coefficient field is invalid.
    """
    _validate_coefficient(coefficient)

    ny, nx = coefficient.shape
    grid = DiffusionGrid(nx=nx, ny=ny)

    num_unknowns_x = nx - 2
    num_unknowns_y = ny - 2
    num_unknowns = num_unknowns_x * num_unknowns_y

    matrix = np.zeros((num_unknowns, num_unknowns), dtype=np.float64)
    rhs = np.full(num_unknowns, forcing, dtype=np.float64)

    for row in range(1, ny - 1):
        for col in range(1, nx - 1):
            center_index = _interior_index(
                row=row,
                col=col,
                nx=nx,
            )

            _add_stencil_entries(
                matrix=matrix,
                rhs=rhs,
                coefficient=coefficient,
                grid=grid,
                boundary_conditions=boundary_conditions,
                row=row,
                col=col,
                center_index=center_index,
            )

    interior_solution = np.linalg.solve(matrix, rhs)

    solution = np.zeros((ny, nx), dtype=np.float64)
    solution = apply_dirichlet_boundary(
        solution=solution,
        boundary_conditions=boundary_conditions,
    )

    for row in range(1, ny - 1):
        for col in range(1, nx - 1):
            index = _interior_index(row=row, col=col, nx=nx)
            solution[row, col] = interior_solution[index]

    return solution


def flatten_solution(solution: np.ndarray) -> np.ndarray:
    """Flatten a two-dimensional solution field.

    Parameters
    ----------
    solution:
        Solution array with shape ``(ny, nx)``.

    Returns
    -------
    np.ndarray
        Flattened solution vector.
    """
    return np.asarray(solution, dtype=np.float32).reshape(-1)


def _validate_coefficient(coefficient: np.ndarray) -> None:
    """Validate the coefficient field."""
    if coefficient.ndim != 2:
        raise ValueError(
            "coefficient must be two-dimensional with shape (ny, nx), "
            f"got shape {coefficient.shape}."
        )

    if coefficient.shape[0] < 4 or coefficient.shape[1] < 4:
        raise ValueError("coefficient grid must be at least 4 by 4.")

    if np.any(coefficient <= 0.0):
        raise ValueError("coefficient must be strictly positive.")


def _interior_index(row: int, col: int, nx: int) -> int:
    """Map an interior grid point to a linear system index."""
    return (row - 1) * (nx - 2) + (col - 1)


def _harmonic_mean(left_value: float, right_value: float) -> float:
    """Compute the harmonic mean of two positive values."""
    return 2.0 * left_value * right_value / (left_value + right_value)


def _add_stencil_entries(
    matrix: np.ndarray,
    rhs: np.ndarray,
    coefficient: np.ndarray,
    grid: DiffusionGrid,
    boundary_conditions: DiffusionBoundaryConditions,
    row: int,
    col: int,
    center_index: int,
) -> None:
    """Add finite-difference stencil entries for one interior grid point."""
    center_value = coefficient[row, col]

    east = _harmonic_mean(center_value, coefficient[row, col + 1])
    west = _harmonic_mean(center_value, coefficient[row, col - 1])
    north = _harmonic_mean(center_value, coefficient[row + 1, col])
    south = _harmonic_mean(center_value, coefficient[row - 1, col])

    east_weight = east / grid.hx**2
    west_weight = west / grid.hx**2
    north_weight = north / grid.hy**2
    south_weight = south / grid.hy**2

    diagonal = east_weight + west_weight + north_weight + south_weight
    matrix[center_index, center_index] = diagonal

    _add_neighbor_or_boundary(
        matrix=matrix,
        rhs=rhs,
        center_index=center_index,
        neighbor_row=row,
        neighbor_col=col + 1,
        weight=east_weight,
        nx=grid.nx,
        ny=grid.ny,
        boundary_value=boundary_conditions.right,
    )
    _add_neighbor_or_boundary(
        matrix=matrix,
        rhs=rhs,
        center_index=center_index,
        neighbor_row=row,
        neighbor_col=col - 1,
        weight=west_weight,
        nx=grid.nx,
        ny=grid.ny,
        boundary_value=boundary_conditions.left,
    )
    _add_neighbor_or_boundary(
        matrix=matrix,
        rhs=rhs,
        center_index=center_index,
        neighbor_row=row + 1,
        neighbor_col=col,
        weight=north_weight,
        nx=grid.nx,
        ny=grid.ny,
        boundary_value=boundary_conditions.top,
    )
    _add_neighbor_or_boundary(
        matrix=matrix,
        rhs=rhs,
        center_index=center_index,
        neighbor_row=row - 1,
        neighbor_col=col,
        weight=south_weight,
        nx=grid.nx,
        ny=grid.ny,
        boundary_value=boundary_conditions.bottom,
    )


def _add_neighbor_or_boundary(
    matrix: np.ndarray,
    rhs: np.ndarray,
    center_index: int,
    neighbor_row: int,
    neighbor_col: int,
    weight: float,
    nx: int,
    ny: int,
    boundary_value: float,
) -> None:
    """Add either an interior neighbor coefficient or boundary contribution."""
    is_boundary = (
        neighbor_row == 0
        or neighbor_row == ny - 1
        or neighbor_col == 0
        or neighbor_col == nx - 1
    )

    if is_boundary:
        rhs[center_index] += weight * boundary_value
        return

    neighbor_index = _interior_index(
        row=neighbor_row,
        col=neighbor_col,
        nx=nx,
    )
    matrix[center_index, neighbor_index] = -weight