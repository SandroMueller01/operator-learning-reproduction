"""Reproduce the paper's Fig. 1/2/3 layout: one figure per PDE, one subplot
per (coefficient, dimension) combo, up to 6 overlaid architecture lines per
subplot (ReLU/ELU/tanh x 4x40/10x100), geometric mean +/- geometric std,
log-log axes, m^-1 reference line.

Existing plot_error_vs_m.py stays as-is for single-run debug plots (one
CSV -> one line); this module is the multi-CSV, multi-panel counterpart
needed to actually match the paper's figures.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless/non-interactive backend for scripted plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ol_reproduction.plotting.aggregation import aggregate_geometric

REQUIRED_COLUMNS = {"problem", "model", "m_train", "relative_test_error"}

PROBLEM_NAME_PATTERN = re.compile(r"^(?P<pde>.+)_(?P<coefficient>affine|log)_d(?P<dimension>\d+)$")

# Paper Fig. 1/2 subplot ordering: affine d=4, affine d=8, log d=4, log d=8.
DEFAULT_SUBPLOT_ORDER = (
    ("affine", 4),
    ("affine", 8),
    ("log", 4),
    ("log", 8),
)


def load_and_tag_metrics(metrics_paths: list[str | Path]) -> pd.DataFrame:
    """Load and concatenate metrics CSVs, tagging each row with the
    (coefficient, dimension) case parsed from its "problem" column
    (e.g. "diffusion_affine_d4" -> coefficient="affine", dimension=4).

    Parameters
    ----------
    metrics_paths:
        Paths to per-(case, architecture) metrics CSV files, as produced by
        scripts/run_pytorch_sweep.py / scripts/run_jax_sweep.py.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe with added "coefficient"/"dimension" columns.

    Raises
    ------
    FileNotFoundError
        If any path does not exist.
    ValueError
        If a CSV is missing required columns, or its "problem" value
        doesn't match the expected "<pde>_<affine|log>_d<dimension>" pattern.
    """
    frames = []

    for path in metrics_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metrics file does not exist: {path}")

        frame = pd.read_csv(path)
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_columns)}"
            )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    parsed = combined["problem"].apply(_parse_problem_name)
    combined["coefficient"] = [p[0] for p in parsed]
    combined["dimension"] = [p[1] for p in parsed]

    return combined


def _parse_problem_name(problem: str) -> tuple[str, int]:
    match = PROBLEM_NAME_PATTERN.match(str(problem))
    if match is None:
        raise ValueError(
            f"problem={problem!r} does not match the expected "
            "'<pde>_<affine|log>_d<dimension>' pattern."
        )
    return match.group("coefficient"), int(match.group("dimension"))


def plot_paper_figure(
    metrics_paths: list[str | Path],
    output_path: str | Path,
    pde_title: str,
    subplot_order: tuple[tuple[str, int], ...] = DEFAULT_SUBPLOT_ORDER,
    reference_slope: float = -1.0,
) -> None:
    """Build a paper-style Fig. 1/2/3 figure from a set of sweep metrics CSVs.

    Parameters
    ----------
    metrics_paths:
        Paths to per-(case, architecture) metrics CSVs (one file per
        coefficient x dimension x architecture combo actually run).
    output_path:
        Where to save the figure.
    pde_title:
        Figure suptitle, e.g. "Elliptic diffusion equation".
    subplot_order:
        Ordered (coefficient, dimension) pairs, one per subplot. Any pair
        with no matching data is skipped (its subplot is left empty with a
        note), so this can be called before every case has been run.
    reference_slope:
        Log-log slope of the dashed reference line (paper: -1, i.e. m^-1).
    """
    data_frame = load_and_tag_metrics(metrics_paths)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, len(subplot_order), figsize=(4.2 * len(subplot_order), 4.2), sharey=True)
    if len(subplot_order) == 1:
        axes = [axes]

    for axis, (coefficient, dimension) in zip(axes, subplot_order):
        subset = data_frame[
            (data_frame["coefficient"] == coefficient) & (data_frame["dimension"] == dimension)
        ]

        subplot_title = f"{coefficient} coeff., d={dimension}"

        if subset.empty:
            axis.set_title(f"{subplot_title}\n(no data yet)")
            axis.set_xlabel(r"$m$")
            continue

        grouped = aggregate_geometric(subset, ["model", "m_train"], "relative_test_error")

        for model_name in sorted(grouped["model"].unique()):
            model_frame = grouped[grouped["model"] == model_name]
            axis.loglog(model_frame["m_train"], model_frame["geometric_mean"], marker="o", markersize=3, label=model_name)
            axis.fill_between(
                model_frame["m_train"], model_frame["band_lower"], model_frame["band_upper"], alpha=0.12
            )

        reference_x, reference_y = _build_reference_line(grouped, reference_slope)
        axis.loglog(reference_x, reference_y, linestyle="--", color="black", linewidth=1, label=rf"$m^{{{reference_slope:g}}}$")

        axis.set_title(subplot_title)
        axis.set_xlabel(r"Number of training samples $m$")
        axis.grid(True, which="both", linestyle=":", linewidth=0.6)

    axes[0].set_ylabel(r"Average relative $L^2_\mu(X;\widetilde{Y})$ error")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 7), bbox_to_anchor=(0.5, -0.05))
    figure.suptitle(pde_title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _build_reference_line(grouped_frame: pd.DataFrame, slope: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference line anchored at the largest m using the smallest
    geometric-mean error observed there."""
    m_values = np.sort(grouped_frame["m_train"].unique()).astype(float)
    max_m = m_values[-1]

    anchor_error = grouped_frame[grouped_frame["m_train"] == max_m]["geometric_mean"].min()
    reference_y = anchor_error * (m_values / max_m) ** slope

    return m_values, reference_y
