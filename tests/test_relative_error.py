"""Tests for relative error metrics."""

from __future__ import annotations

import numpy as np

from ol_reproduction.evaluation.relative_error import relative_l2_error


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