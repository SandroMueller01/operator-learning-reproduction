"""Plot activation function comparison results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "activation",
    "m_train",
    "relative_test_error",
}


def plot_activation_comparison(
    metrics_paths: list[str | Path],
    output_path: str | Path,
    title: str | None = None,
    reference_slope: float = -1.0,
) -> None:
    """Plot relative test error for multiple activation functions.

    Parameters
    ----------
    metrics_paths:
        List of metrics CSV paths. Each CSV should contain at least the columns
        ``activation``, ``m_train``, and ``relative_test_error``.
    output_path:
        Path where the figure should be saved.
    title:
        Optional plot title.
    reference_slope:
        Slope of the reference line shown on the log-log plot. A value of
        ``-1.0`` corresponds to an ``m^{-1}`` reference rate.

    Raises
    ------
    ValueError
        If no metrics paths are provided or if required columns are missing.
    FileNotFoundError
        If one of the metrics files does not exist.
    """
    if not metrics_paths:
        raise ValueError("At least one metrics CSV path must be provided.")

    frames = [_load_metrics_frame(path) for path in metrics_paths]
    combined_frame = pd.concat(frames, ignore_index=True)

    _validate_metrics_frame(combined_frame)

    grouped = _aggregate_by_activation_and_m(combined_frame)

    figure, axis = plt.subplots(figsize=(6.5, 4.5))

    for activation in sorted(grouped["activation"].unique()):
        activation_frame = grouped[grouped["activation"] == activation]

        m_values = activation_frame["m_train"].to_numpy()
        mean_error = activation_frame["mean_error"].to_numpy()
        std_error = activation_frame["std_error"].to_numpy()

        axis.loglog(
            m_values,
            mean_error,
            marker="o",
            label=activation,
        )

        lower = np.maximum(mean_error - std_error, 1.0e-16)
        upper = mean_error + std_error

        axis.fill_between(
            m_values,
            lower,
            upper,
            alpha=0.15,
        )

    reference_x, reference_y = _build_reference_line(
        grouped_frame=grouped,
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _load_metrics_frame(path: str | Path) -> pd.DataFrame:
    """Load one metrics CSV file.

    Parameters
    ----------
    path:
        Path to metrics CSV.

    Returns
    -------
    pd.DataFrame
        Loaded metrics dataframe.
    """
    metrics_path = Path(path)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_path}")

    data_frame = pd.read_csv(metrics_path)
    _validate_metrics_frame(data_frame)

    return data_frame


def _validate_metrics_frame(data_frame: pd.DataFrame) -> None:
    """Validate metrics dataframe.

    Parameters
    ----------
    data_frame:
        Metrics dataframe.

    Raises
    ------
    ValueError
        If the dataframe is invalid.
    """
    missing_columns = REQUIRED_COLUMNS.difference(data_frame.columns)

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


def _aggregate_by_activation_and_m(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by activation and training size.

    Parameters
    ----------
    data_frame:
        Raw metrics dataframe.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe.
    """
    grouped = (
        data_frame.groupby(["activation", "m_train"], as_index=False)
        .agg(
            mean_error=("relative_test_error", "mean"),
            std_error=("relative_test_error", "std"),
        )
        .sort_values(["activation", "m_train"])
    )

    grouped["std_error"] = grouped["std_error"].fillna(0.0)

    return grouped


def _build_reference_line(
    grouped_frame: pd.DataFrame,
    slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a reference convergence line.

    The line is anchored at the largest training size using the smallest
    observed mean error at that training size.

    Parameters
    ----------
    grouped_frame:
        Aggregated metrics dataframe.
    slope:
        Reference log-log slope.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Reference x and y values.
    """
    m_values = np.sort(grouped_frame["m_train"].unique()).astype(float)
    max_m = m_values[-1]

    largest_m_frame = grouped_frame[grouped_frame["m_train"] == max_m]
    anchor_error = largest_m_frame["mean_error"].min()

    reference_y = anchor_error * (m_values / max_m) ** slope

    return m_values, reference_y