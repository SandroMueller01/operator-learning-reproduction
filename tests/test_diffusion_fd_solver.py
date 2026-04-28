"""Tests for the finite-difference diffusion solver."""

from __future__ import annotations

import numpy as np

from ol_reproduction.coefficients.affine import affine_coefficient
from ol_reproduction.pde.diffusion.fd_solver import (
    DiffusionBoundaryConditions,
    DiffusionGrid,
    create_unit_square_grid,
    flatten_solution,
    solve_diffusion_fd,
)


def test_create_unit_square_grid_shape() -> None:
    """The generated grid should have shape (ny, nx)."""
    grid = DiffusionGrid(nx=16, ny=12)

    z1, z2 = create_unit_square_grid(grid)

    assert z1.shape == (12, 16)
    assert z2.shape == (12, 16)


def test_solve_diffusion_fd_shape() -> None:
    """The solver should return a solution with the coefficient shape."""
    grid = DiffusionGrid(nx=12, ny=12)
    z1, _ = create_unit_square_grid(grid)
    parameters = np.zeros(4)
    coefficient = affine_coefficient(z1=z1, parameters=parameters)

    solution = solve_diffusion_fd(
        coefficient=coefficient,
        forcing=10.0,
        boundary_conditions=DiffusionBoundaryConditions(),
    )

    assert solution.shape == coefficient.shape


def test_solve_diffusion_fd_boundary_values() -> None:
    """The solver should enforce Dirichlet boundary values."""
    grid = DiffusionGrid(nx=12, ny=12)
    z1, _ = create_unit_square_grid(grid)
    parameters = np.zeros(4)
    coefficient = affine_coefficient(z1=z1, parameters=parameters)

    boundary_conditions = DiffusionBoundaryConditions(
        bottom=0.5,
        top=0.0,
        left=0.0,
        right=0.0,
    )

    solution = solve_diffusion_fd(
        coefficient=coefficient,
        forcing=10.0,
        boundary_conditions=boundary_conditions,
    )

    np.testing.assert_allclose(solution[0, 1:-1], 0.5)
    np.testing.assert_allclose(solution[-1, 1:-1], 0.0)
    np.testing.assert_allclose(solution[1:-1, 0], 0.0)
    np.testing.assert_allclose(solution[1:-1, -1], 0.0)


def test_flatten_solution_shape() -> None:
    """Flattening should return a one-dimensional vector."""
    solution = np.zeros((10, 12))

    flattened = flatten_solution(solution)

    assert flattened.shape == (120,)
    assert flattened.dtype == np.float32


def test_solve_diffusion_fd_rejects_negative_coefficient() -> None:
    """The solver should reject non-positive coefficient fields."""
    coefficient = -np.ones((8, 8))

    try:
        solve_diffusion_fd(
            coefficient=coefficient,
            forcing=10.0,
            boundary_conditions=DiffusionBoundaryConditions(),
        )
    except ValueError as error:
        assert "coefficient must be strictly positive" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")