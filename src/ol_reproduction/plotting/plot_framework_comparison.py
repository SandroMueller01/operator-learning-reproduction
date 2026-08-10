"""Plot framework comparison results for PyTorch and JAX."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless/non-interactive backend for scripted plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ol_reproduction.plotting.aggregation import aggregate_geometric


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


def _plot_relative_error(
    data_frame: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot relative test error for each framework.

    Aggregation is geometric mean / geometric std (paper's protocol, see
    ol_reproduction.plotting.aggregation), not arithmetic.

    Parameters
    ----------
    data_frame:
        Combined metrics dataframe.
    output_path:
        Figure output path.
    title:
        Plot title.
    """
    grouped = aggregate_geometric(data_frame, ["framework", "m_train"], "relative_test_error")

    figure, axis = plt.subplots(figsize=(6.5, 4.5))

    for framework in sorted(grouped["framework"].unique()):
        framework_frame = grouped[grouped["framework"] == framework]

        axis.loglog(
            framework_frame["m_train"],
            framework_frame["geometric_mean"],
            marker="o",
            label=framework,
        )

        axis.fill_between(
            framework_frame["m_train"],
            framework_frame["band_lower"],
            framework_frame["band_upper"],
            alpha=0.15,
        )

    reference_x, reference_y = _build_reference_line(grouped, "geometric_mean")

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
    """Plot training time for each framework (geometric mean/std).

    Parameters
    ----------
    data_frame:
        Combined metrics dataframe.
    output_path:
        Figure output path.
    title:
        Plot title.
    """
    grouped = aggregate_geometric(data_frame, ["framework", "m_train"], "training_time_sec")

    figure, axis = plt.subplots(figsize=(6.5, 4.5))

    for framework in sorted(grouped["framework"].unique()):
        framework_frame = grouped[grouped["framework"] == framework]

        axis.plot(
            framework_frame["m_train"],
            framework_frame["geometric_mean"],
            marker="o",
            label=framework,
        )

        axis.fill_between(
            framework_frame["m_train"],
            framework_frame["band_lower"],
            framework_frame["band_upper"],
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
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an m^{-1} reference line for the error plot.

    The reference line is anchored at the largest training size using the
    smallest aggregated value observed there.

    Parameters
    ----------
    grouped_frame:
        Aggregated metrics dataframe (from aggregate_geometric).
    value_column:
        Column to anchor the reference line to (e.g. "geometric_mean").

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Reference x and y values.
    """
    m_values = np.sort(grouped_frame["m_train"].unique()).astype(float)
    max_m = m_values[-1]

    largest_m_frame = grouped_frame[grouped_frame["m_train"] == max_m]
    anchor_error = largest_m_frame[value_column].min()

    reference_y = anchor_error * (m_values / max_m) ** (-1.0)

    return m_values, reference_y