"""Tests for the affine diffusion coefficient."""

from __future__ import annotations

import numpy as np

from ol_reproduction.coefficients.affine import affine_coefficient


def test_affine_coefficient_shape() -> None:
    """Coefficient should have the same shape as the spatial grid."""
    grid = np.linspace(0.0, 1.0, 16)
    z1, _ = np.meshgrid(grid, grid, indexing="xy")
    parameters = np.array([0.1, -0.2, 0.3, -0.4])

    coefficient = affine_coefficient(
        z1=z1,
        parameters=parameters,
    )

    assert coefficient.shape == z1.shape


def test_affine_coefficient_base_value_for_zero_parameters() -> None:
    """Zero parameters should return the constant base coefficient."""
    grid = np.linspace(0.0, 1.0, 16)
    z1, _ = np.meshgrid(grid, grid, indexing="xy")
    parameters = np.zeros(4)

    coefficient = affine_coefficient(
        z1=z1,
        parameters=parameters,
        base_value=2.62,
    )

    expected = np.full_like(z1, 2.62, dtype=np.float64)
    np.testing.assert_allclose(coefficient, expected)


def test_affine_coefficient_positive_for_extreme_parameters() -> None:
    """Coefficient should stay positive for parameters in [-1, 1]^d."""
    grid = np.linspace(0.0, 1.0, 64)
    z1, _ = np.meshgrid(grid, grid, indexing="xy")

    for sign in (-1.0, 1.0):
        parameters = sign * np.ones(8)
        coefficient = affine_coefficient(
            z1=z1,
            parameters=parameters,
        )

        assert np.min(coefficient) > 0.0


def test_affine_coefficient_rejects_non_vector_parameters() -> None:
    """Coefficient should reject parameter arrays that are not 1D."""
    grid = np.linspace(0.0, 1.0, 8)
    z1, _ = np.meshgrid(grid, grid, indexing="xy")
    parameters = np.zeros((2, 2))

    try:
        affine_coefficient(z1=z1, parameters=parameters)
    except ValueError as error:
        assert "parameters must be one-dimensional" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")
