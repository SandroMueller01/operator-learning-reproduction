"""Utilities for estimating convergence rates from experiment metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


required_columns = {
    "m_train",
    "relative_test_error",
}


def estimate_loglog_slope(
    m_values: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float]:
    """Estimate the log-log convergence slope.

    This fits the model

    .. math::

        \\log(e_m) = \\alpha \\log(m) + \\beta,

    where ``alpha`` is the empirical convergence slope.

    Parameters
    ----------
    m_values:
        Training sizes.
    errors:
        Relative test errors.

    Returns
    -------
    tuple[float, float]
        Estimated slope ``alpha`` and intercept ``beta``.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    m_values = np.asarray(m_values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)

    _validate_slope_inputs(m_values=m_values, errors=errors)

    log_m = np.log(m_values)
    log_errors = np.log(errors)

    slope, intercept = np.polyfit(log_m, log_errors, deg=1)

    return float(slope), float(intercept)


def estimate_slope_from_metrics(
    metrics_path: str | Path,
) -> dict[str, float]:
    """Estimate convergence slope from a metrics CSV file.

    If multiple seeds are present, errors are averaged by training size before
    fitting the slope.

    Parameters
    ----------
    metrics_path:
        Path to metrics CSV.

    Returns
    -------
    dict[str, float]
        Summary containing slope, intercept, and number of training sizes.

    Raises
    ------
    FileNotFoundError
        If the metrics file does not exist.
    ValueError
        If the metrics file is invalid.
    """
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_path}")

    data_frame = pd.read_csv(metrics_path)
    _validate_metrics_frame(data_frame)

    grouped = (
        data_frame.groupby("m_train", as_index=False)
        .agg(mean_error=("relative_test_error", "mean"))
        .sort_values("m_train")
    )

    slope, intercept = estimate_loglog_slope(
        m_values=grouped["m_train"].to_numpy(),
        errors=grouped["mean_error"].to_numpy(),
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "num_points": float(len(grouped)),
        "min_m": float(grouped["m_train"].min()),
        "max_m": float(grouped["m_train"].max()),
    }


def _validate_slope_inputs(
    m_values: np.ndarray,
    errors: np.ndarray,
) -> None:
    """Validate inputs for slope estimation.

    Parameters
    ----------
    m_values:
        Training sizes.
    errors:
        Relative test errors.
    """
    if m_values.ndim != 1:
        raise ValueError("m_values must be one-dimensional.")

    if errors.ndim != 1:
        raise ValueError("errors must be one-dimensional.")

    if m_values.shape[0] != errors.shape[0]:
        raise ValueError("m_values and errors must have the same length.")

    if m_values.shape[0] < 2:
        raise ValueError("At least two data points are required.")

    if np.any(m_values <= 0.0):
        raise ValueError("m_values must be strictly positive.")

    if np.any(errors <= 0.0):
        raise ValueError("errors must be strictly positive.")


def _validate_metrics_frame(data_frame: pd.DataFrame) -> None:
    """Validate metrics dataframe.

    Parameters
    ----------
    data_frame:
        Metrics dataframe.
    """
    missing_columns = required_columns.difference(data_frame.columns)

    if missing_columns:
        raise ValueError(
            "Metrics CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data_frame.empty:
        raise ValueError("Metrics CSV is empty.")

    if (data_frame["m_train"] <= 0).any():
        raise ValueError("m_train must be strictly positive.")

    if (data_frame["relative_test_error"] <= 0.0).any():
        raise ValueError("relative_test_error must be strictly positive.")