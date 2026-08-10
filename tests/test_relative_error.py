"""Tests for relative error metrics."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from ol_reproduction.evaluation.relative_error import (
    relative_l2_error,
    relative_l2_error_mass_weighted,
)


def test_relative_l2_error_zero_for_exact_prediction() -> None:
    """Relative error should be zero when prediction equals truth."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = y_true.copy()

    error = relative_l2_error(y_true=y_true, y_pred=y_pred)

    assert error == 0.0


def test_relative_l2_error_positive_for_inexact_prediction() -> None:
    """Relative error should be positive for an imperfect prediction."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.zeros_like(y_true)

    error = relative_l2_error(y_true=y_true, y_pred=y_pred)

    assert error > 0.0


def test_relative_l2_error_rejects_shape_mismatch() -> None:
    """Relative error should reject mismatched shapes."""
    y_true = np.zeros((2, 3))
    y_pred = np.zeros((2, 4))

    try:
        relative_l2_error(y_true=y_true, y_pred=y_pred)
    except ValueError as error:
        assert "same shape" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")


def test_mass_weighted_error_zero_for_exact_prediction() -> None:
    mass_matrix = sp.identity(3, format="csr")
    y_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_pred = y_true.copy()

    error = relative_l2_error_mass_weighted(
        y_true=y_true, y_pred=y_pred, mass_matrix=mass_matrix
    )

    assert error == pytest.approx(0.0, abs=1e-12)


def test_mass_weighted_error_matches_identity_mass_matrix_case() -> None:
    """With an identity mass matrix and uniform weights, this should match
    the unweighted relative_l2_error exactly (both reduce to plain
    Euclidean sum-of-squares)."""
    mass_matrix = sp.identity(4, format="csr")
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(5, 4))
    y_pred = rng.normal(size=(5, 4))

    plain = relative_l2_error(y_true=y_true, y_pred=y_pred)
    mass_weighted = relative_l2_error_mass_weighted(
        y_true=y_true, y_pred=y_pred, mass_matrix=mass_matrix
    )

    assert mass_weighted == pytest.approx(plain, rel=1e-8)


def test_mass_weighted_error_uses_off_diagonal_mass_matrix_correctly() -> None:
    """Hand-computed reference for a non-trivial (non-diagonal) mass matrix,
    mimicking a real FEM mass matrix's off-diagonal coupling between
    neighboring basis functions."""
    mass_matrix = sp.csr_matrix(
        np.array([[2.0, 1.0], [1.0, 2.0]])
    )
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.0, 0.0], [0.0, 0.0]])

    # v^T M v for v=[1,0]: [1,0] @ [[2,1],[1,2]] @ [1,0]^T = 2
    # v^T M v for v=[0,1]: [0,1] @ [[2,1],[1,2]] @ [0,1]^T = 2
    # difference == y_true here since y_pred is all zeros.
    expected = np.sqrt((2.0 + 2.0) / (2.0 + 2.0))

    error = relative_l2_error_mass_weighted(
        y_true=y_true, y_pred=y_pred, mass_matrix=mass_matrix
    )

    assert error == pytest.approx(expected, rel=1e-10)


def test_mass_weighted_error_applies_parametric_weights() -> None:
    mass_matrix = sp.identity(2, format="csr")
    y_true = np.array([[1.0, 0.0], [1.0, 0.0]])
    y_pred = np.array([[0.0, 0.0], [0.0, 0.0]])

    # Weighting the second (identical) sample twice as heavily shouldn't
    # change the ratio here since both samples are identical.
    error_uniform = relative_l2_error_mass_weighted(
        y_true=y_true, y_pred=y_pred, mass_matrix=mass_matrix
    )
    error_weighted = relative_l2_error_mass_weighted(
        y_true=y_true,
        y_pred=y_pred,
        mass_matrix=mass_matrix,
        parametric_weights=np.array([1.0, 2.0]),
    )

    assert error_uniform == pytest.approx(error_weighted, rel=1e-10)


def test_mass_weighted_error_rejects_dimension_mismatch() -> None:
    mass_matrix = sp.identity(5, format="csr")
    y_true = np.zeros((2, 3))
    y_pred = np.zeros((2, 3))

    with pytest.raises(ValueError):
        relative_l2_error_mass_weighted(
            y_true=y_true, y_pred=y_pred, mass_matrix=mass_matrix
        )