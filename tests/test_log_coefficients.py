"""Tests for the log-transformed coefficient a_{2,d} (paper eq. B.2)."""

from __future__ import annotations

import numpy as np
import pytest

from ol_reproduction.coefficients.log_transformed import (
    BETA,
    BETA_P,
    log_transformed_coefficient,
)


def test_beta_constants() -> None:
    """beta_c = 1/8, beta_p = max(1, 2 beta_c) = 1, beta = beta_c / beta_p."""
    assert BETA_P == pytest.approx(1.0)
    assert BETA == pytest.approx(0.125)


def test_matches_hand_computed_reference_d1() -> None:
    """d=1: a2 = exp(1 + x1 * sqrt(sqrt(pi beta) / 2))."""
    z1 = np.array([0.3])
    parameters = np.array([0.5])

    expected = np.exp(1.0 + 0.5 * np.sqrt(np.sqrt(np.pi * BETA) / 2.0))
    got = log_transformed_coefficient(z1=z1, parameters=parameters)

    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_matches_hand_computed_reference_d3() -> None:
    """d=3 exercises both the even (sin) and odd (cos) theta_j branches."""
    z1 = np.array([0.7])
    parameters = np.array([0.4, -0.2, 0.9])

    log_coefficient = 1.0 + 0.4 * np.sqrt(np.sqrt(np.pi * BETA) / 2.0)

    half_j = 1  # floor(2 / 2) == floor(3 / 2) == 1
    zeta_j = np.sqrt(np.sqrt(np.pi * BETA)) * np.exp(
        -((half_j * np.pi * BETA) ** 2) / 8.0
    )

    theta_2 = np.sin(half_j * np.pi * z1 / BETA_P)  # j=2 is even
    log_coefficient = log_coefficient + (-0.2) * zeta_j * theta_2

    theta_3 = np.cos(half_j * np.pi * z1 / BETA_P)  # j=3 is odd
    log_coefficient = log_coefficient + 0.9 * zeta_j * theta_3

    expected = np.exp(log_coefficient)
    got = log_transformed_coefficient(z1=z1, parameters=parameters)

    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_depends_only_on_z1() -> None:
    """The coefficient must not depend on z2, matching the paper's formula."""
    z1 = np.full((4, 4), 0.42)
    parameters = np.array([0.1, -0.3, 0.6, 0.2])

    result = log_transformed_coefficient(z1=z1, parameters=parameters)

    assert np.allclose(result, result.flat[0])


def test_output_shape_matches_z1() -> None:
    z1 = np.linspace(0.0, 1.0, 16).reshape(4, 4)
    parameters = np.array([0.1, 0.2, -0.3, 0.4])

    result = log_transformed_coefficient(z1=z1, parameters=parameters)

    assert result.shape == z1.shape


def test_positive_coefficient() -> None:
    """a_{2,d} is exp(...), so it must always be strictly positive."""
    rng = np.random.default_rng(0)
    z1 = rng.uniform(0.0, 1.0, size=(8, 8))
    parameters = rng.uniform(-1.0, 1.0, size=8)

    result = log_transformed_coefficient(z1=z1, parameters=parameters)

    assert np.all(result > 0.0)


def test_rejects_non_one_dimensional_parameters() -> None:
    z1 = np.array([0.5])

    with pytest.raises(ValueError):
        log_transformed_coefficient(z1=z1, parameters=np.array([[0.1, 0.2]]))
