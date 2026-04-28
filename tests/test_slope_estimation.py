"""Tests for empirical slope estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ol_reproduction.evaluation.slope_estimation import (
    estimate_loglog_slope,
    estimate_slope_from_metrics,
)


def test_estimate_loglog_slope_for_inverse_rate() -> None:
    """Slope should be close to -1 for errors proportional to 1 / m."""
    m_values = np.array([10, 20, 50, 100], dtype=np.float64)
    errors = 1.0 / m_values

    slope, _ = estimate_loglog_slope(
        m_values=m_values,
        errors=errors,
    )

    np.testing.assert_allclose(slope, -1.0, atol=1.0e-12)


def test_estimate_loglog_slope_rejects_negative_error() -> None:
    """Slope estimation should reject non-positive errors."""
    m_values = np.array([10, 20, 50], dtype=np.float64)
    errors = np.array([1.0, -0.5, 0.2], dtype=np.float64)

    try:
        estimate_loglog_slope(m_values=m_values, errors=errors)
    except ValueError as error:
        assert "errors must be strictly positive" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")


def test_estimate_slope_from_metrics(tmp_path) -> None:
    """Slope should be estimated from a metrics CSV file."""
    metrics_path = tmp_path / "metrics.csv"

    data_frame = pd.DataFrame(
        {
            "m_train": [10, 20, 50, 100],
            "relative_test_error": [0.1, 0.05, 0.02, 0.01],
        }
    )
    data_frame.to_csv(metrics_path, index=False)

    summary = estimate_slope_from_metrics(metrics_path=metrics_path)

    np.testing.assert_allclose(summary["slope"], -1.0, atol=1.0e-12)
    assert summary["num_points"] == 4.0
    assert summary["min_m"] == 10.0
    assert summary["max_m"] == 100.0