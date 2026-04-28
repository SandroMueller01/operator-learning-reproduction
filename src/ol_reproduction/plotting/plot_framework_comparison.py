"""Plot framework comparison results for PyTorch and JAX."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "framework",
    "m_train",
    "relative_test_error",
    "training_time_sec",
}


def plot_framework_comparison(
    pytorch_metrics_path: str | Path,
    jax_metrics_path: str | Path,
    output_dir: str | Path,
    experiment_name: str,
) -> None:
    """Create framework comparison plots.

    Parameters
    ----------
    pytorch_metrics_path:
        Path to the PyTorch metrics CSV file.
    jax_metrics_path:
        Path to the JAX metrics CSV file.
    output_dir:
        Directory where the figures should be saved.
    experiment_name:
        Name used in output file names and plot titles.
    """
    pytorch_frame = _load_metrics_frame(pytorch_metrics_path)
    jax_frame = _load_metrics_frame(jax_metrics_path)

    combined_frame = pd.concat(
        [pytorch_frame, jax_frame],
        ignore_index=True,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    error_output_path = output_dir / f"{experiment_name}_framework_error.png"
    time_output_path = output_dir / f"{experiment_name}_framework_time.png"

    _plot_relative_error(
        data_frame=combined_frame,
        output_path=error_output_path,
        title=f"{experiment_name}: Relative Test Error",
    )
    _plot_training_time(
        data_frame=combined_frame,
        output_path=time_output_path,
        title=f"{experiment_name}: Training Time",
    )


def _load_metrics_frame(path: str | Path) -> pd.DataFrame:
    """Load and validate one metrics CSV file.

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
    """Validate required metrics columns.

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

    if (data_frame["training_time_sec"] <= 0.0).any():
        raise ValueError("training_time_sec must be strictly positive.")


def _aggregate_by_framework_and_m(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by framework and training size.

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
        data_frame.groupby(["framework", "m_train"], as_index=False)
        .agg(
            mean_error=("relative_test_error", "mean"),
            std_error=("relative_test_error", "std"),
            mean_time=("training_time_sec", "mean"),
            std_time=("training_time_sec", "std"),
        )
        .sort_values(["framework", "m_train"])
    )

    grouped["std_error"] = grouped["std_error"].fillna(0.0)
    grouped["std_time"] = grouped["std_time"].fillna(0.0)

    return grouped


def _plot_relative_error(
    data_frame: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot relative test error for each framework.

    Parameters
    ----------
    data_frame:
        Combined metrics dataframe.
    output_path:
        Figure output path.
    title:
        Plot title.
    """
    grouped = _aggregate_by_framework_and_m(data_frame)

    figure, axis = plt.subplots(figsize=(6.5, 4.5))

    for framework in sorted(grouped["framework"].unique()):
        framework_frame = grouped[grouped["framework"] == framework]

        m_values = framework_frame["m_train"].to_numpy()
        mean_error = framework_frame["mean_error"].to_numpy()
        std_error = framework_frame["std_error"].to_numpy()

        axis.loglog(
            m_values,
            mean_error,
            marker="o",
            label=framework,
        )

        lower = np.maximum(mean_error - std_error, 1.0e-16)
        upper = mean_error + std_error

        axis.fill_between(
            m_values,
            lower,
            upper,
            alpha=0.15,
        )

    reference_x, reference_y = _build_reference_line(grouped)

    axis.loglog(
        reference_x,
        reference_y,
        linestyle="--",
        label=r"Reference rate $m^{-1}$",
    )

    axis.set_xlabel(r"Number of training samples $m$")
    axis.set_ylabel("Relative test error")
    axis.set_title(title)
    axis.grid(True, which="both", linestyle=":", linewidth=0.8)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _plot_training_time(
    data_frame: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot training time for each framework.

    Parameters
    ----------
    data_frame:
        Combined metrics dataframe.
    output_path:
        Figure output path.
    title:
        Plot title.
    """
    grouped = _aggregate_by_framework_and_m(data_frame)

    figure, axis = plt.subplots(figsize=(6.5, 4.5))

    for framework in sorted(grouped["framework"].unique()):
        framework_frame = grouped[grouped["framework"] == framework]

        m_values = framework_frame["m_train"].to_numpy()
        mean_time = framework_frame["mean_time"].to_numpy()
        std_time = framework_frame["std_time"].to_numpy()

        axis.plot(
            m_values,
            mean_time,
            marker="o",
            label=framework,
        )

        lower = np.maximum(mean_time - std_time, 0.0)
        upper = mean_time + std_time

        axis.fill_between(
            m_values,
            lower,
            upper,
            alpha=0.15,
        )

    axis.set_xlabel(r"Number of training samples $m$")
    axis.set_ylabel("Training time [s]")
    axis.set_title(title)
    axis.grid(True, linestyle=":", linewidth=0.8)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _build_reference_line(
    grouped_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an m^{-1} reference line for the error plot.

    The reference line is anchored at the largest training size using the
    smallest mean error observed there.

    Parameters
    ----------
    grouped_frame:
        Aggregated metrics dataframe.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Reference x and y values.
    """
    m_values = np.sort(grouped_frame["m_train"].unique()).astype(float)
    max_m = m_values[-1]

    largest_m_frame = grouped_frame[grouped_frame["m_train"] == max_m]
    anchor_error = largest_m_frame["mean_error"].min()

    reference_y = anchor_error * (m_values / max_m) ** (-1.0)

    return m_values, reference_y