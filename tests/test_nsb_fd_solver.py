"""Tests for the simplified NSB finite-difference solver."""

from __future__ import annotations

import numpy as np

from ol_reproduction.coefficients.affine import affine_coefficient
from ol_reproduction.pde.navier_stokes_brinkman.fd_solver import (
    NsbGrid,
    create_unit_square_grid,
    flatten_pressure,
    flatten_velocity,
    solve_nsb_fd,
)


def test_nsb_solver_shapes() -> None:
    """NSB solver should return velocity and pressure with correct shapes."""
    grid = NsbGrid(nx=8, ny=8)
    z1, _ = create_unit_square_grid(grid)
    parameters = np.zeros(4)
    viscosity = affine_coefficient(z1=z1, parameters=parameters)

    velocity, pressure = solve_nsb_fd(viscosity=viscosity)

    assert velocity.shape == (8, 8, 2)
    assert pressure.shape == (8, 8)


def test_nsb_solver_flatten_shapes() -> None:
    """Flattened NSB outputs should have expected dimensions."""
    velocity = np.zeros((8, 8, 2))
    pressure = np.zeros((8, 8))

    flattened_velocity = flatten_velocity(velocity)
    flattened_pressure = flatten_pressure(pressure)

    assert flattened_velocity.shape == (128,)
    assert flattened_pressure.shape == (64,)


def test_nsb_pressure_has_zero_mean() -> None:
    """Pressure should be normalized to approximately zero mean."""
    grid = NsbGrid(nx=8, ny=8)
    z1, _ = create_unit_square_grid(grid)
    parameters = np.zeros(4)
    viscosity = affine_coefficient(z1=z1, parameters=parameters)

    _, pressure = solve_nsb_fd(viscosity=viscosity)

    assert abs(float(np.mean(pressure))) < 1.0e-10


def test_nsb_solver_rejects_negative_viscosity() -> None:
    """NSB solver should reject non-positive viscosity."""
    viscosity = -np.ones((8, 8))

    try:
        solve_nsb_fd(viscosity=viscosity)
    except ValueError as error:
        assert "viscosity must be strictly positive" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")