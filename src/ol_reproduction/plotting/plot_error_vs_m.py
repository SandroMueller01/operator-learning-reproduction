"""Plot relative test error against training set size."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless/non-interactive backend for scripted plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ol_reproduction.plotting.aggregation import aggregate_geometric


REQUIRED_COLUMNS = {
    "m_train",
    "relative_test_error",
}


def plot_error_vs_m(
    metrics_path: str | Path,
    output_path: str | Path,
    title: str | None = None,
    reference_slope: float = -1.0,
) -> None:
    """Plot relative test error versus number of training samples.

    Parameters
    ----------
    metrics_path:
        Path to the CSV file containing experiment metrics.
    output_path:
        Path where the figure should be saved.
    title:
        Optional plot title.
    reference_slope:
        Slope of the reference line shown on the log-log plot.
        A value of ``-1.0`` corresponds to an ``m^{-1}`` reference rate.

    Raises
    ------
    FileNotFoundError
        If the metrics file does not exist.
    ValueError
        If the CSV file does not contain the required columns.
    """
    metrics_path = Path(metrics_path)
    output_path = Path(output_path)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_path}")

    data_frame = pd.read_csv(metrics_path)
    _validate_metrics_frame(data_frame)

    grouped = aggregate_geometric(data_frame, ["m_train"], "relative_test_error")

    figure, axis = plt.subplots(figsize=(6.5, 4.5))

    axis.loglog(
        grouped["m_train"],
        grouped["geometric_mean"],
        marker="o",
        label="Measured relative test error (geometric mean)",
    )

    axis.fill_between(
        grouped["m_train"],
        grouped["band_lower"],
        grouped["band_upper"],
        alpha=0.2,
        label=r"$\times / \div$ 1 geometric std",
    )

    reference_x, reference_y = _build_reference_line(
        m_values=grouped["m_train"].to_numpy(),
        errors=grouped["geometric_mean"].to_numpy(),
        slope=reference_slope,
    )

    axis.loglog(
        reference_x,
        reference_y,
        linestyle="--",
        label=rf"Reference rate $m^{{{reference_slope:g}}}$",
    )

    axis.set_xlabel(r"Number of training samples $m$")
    axis.set_ylabel("Relative test error")
    axis.grid(True, which="both", linestyle=":", linewidth=0.8)
    axis.legend()

    if title is not None:
        axis.set_title(title)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _validate_metrics_frame(data_frame: pd.DataFrame) -> None:
    """Validate that the metrics dataframe has the required columns.

    Parameters
    ----------
    data_frame:
        Metrics dataframe.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing_columns = REQUIRED_COLUMNS.difference(data_frame.columns)

    if missing_columns:
        raise ValueError(
            "Metrics CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data_frame.empty:
        raise ValueError("Metrics CSV is empty.")

    if (data_frame["relative_test_error"] <= 0.0).any():
        raise ValueError("relative_test_error must be strictly positive.")

    if (data_frame["m_train"] <= 0).any():
        raise ValueError("m_train must be strictly positive.")


def _build_reference_line(
    m_values: np.ndarray,
    errors: np.ndarray,
    slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a reference convergence line.

    The reference line is anchored at the largest training size.

    Parameters
    ----------
    m_values:
        Training sizes.
    errors:
        Mean relative test errors.
    slope:
        Reference log-log slope.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Reference x and y values.
    """
    sorted_indices = np.argsort(m_values)
    reference_x = m_values[sorted_indices].astype(float)

    anchor_m = reference_x[-1]
    anchor_error = errors[sorted_indices][-1]

    reference_y = anchor_error * (reference_x / anchor_m) ** slope

    return reference_x, reference_y