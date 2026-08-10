"""Tests for the simplified Boussinesq finite-difference solver."""

from __future__ import annotations

import numpy as np

from ol_reproduction.coefficients.affine import affine_coefficient
from ol_reproduction.pde.boussinesq.fd_solver import (
    flatten_boussinesq_pressure,
    flatten_boussinesq_velocity,
    flatten_temperature,
    solve_boussinesq_fd,
)


def test_boussinesq_solver_shapes() -> None:
    """Boussinesq solver should return fields with correct shapes."""
    grid = np.linspace(0.0, 1.0, 8)
    z1, _ = np.meshgrid(grid, grid, indexing="xy")
    parameters = np.zeros(4)

    viscosity = affine_coefficient(z1=z1, parameters=parameters)
    thermal_conductivity = affine_coefficient(z1=z1, parameters=parameters)

    velocity, temperature, pressure = solve_boussinesq_fd(
        viscosity=viscosity,
        thermal_conductivity=thermal_conductivity,
    )

    assert velocity.shape == (8, 8, 2)
    assert temperature.shape == (8, 8)
    assert pressure.shape == (8, 8)


def test_boussinesq_flatten_shapes() -> None:
    """Flattened Boussinesq outputs should have expected dimensions."""
    velocity = np.zeros((8, 8, 2))
    temperature = np.zeros((8, 8))
    pressure = np.zeros((8, 8))

    flattened_velocity = flatten_boussinesq_velocity(velocity)
    flattened_temperature = flatten_temperature(temperature)
    flattened_pressure = flatten_boussinesq_pressure(pressure)

    assert flattened_velocity.shape == (128,)
    assert flattened_temperature.shape == (64,)
    assert flattened_pressure.shape == (64,)


def test_boussinesq_rejects_shape_mismatch() -> None:
    """Solver should reject mismatched coefficient shapes."""
    viscosity = np.ones((8, 8))
    thermal_conductivity = np.ones((7, 8))

    try:
        solve_boussinesq_fd(
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
        )
    except ValueError as error:
        assert "same shape" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")