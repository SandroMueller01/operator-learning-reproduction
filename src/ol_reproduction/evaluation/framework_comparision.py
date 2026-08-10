"""PyTorch vs JAX comparison (Phase 12 final deliverable).

Computes, per training size m, the geometric-mean relative test error and
training time for each framework, plus the JAX/PyTorch ratio of each --
the numeric counterpart to
``ol_reproduction.plotting.plot_framework_comparison`` (which only plots).
Also reports each framework's overall log-log convergence slope via
``ol_reproduction.evaluation.slope_estimation``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ol_reproduction.evaluation.slope_estimation import estimate_slope_from_metrics
from ol_reproduction.plotting.aggregation import aggregate_geometric

REQUIRED_COLUMNS = {"m_train", "relative_test_error", "training_time_sec"}


def compare_frameworks(
    pytorch_metrics_path: str | Path,
    jax_metrics_path: str | Path,
) -> pd.DataFrame:
    """Compare PyTorch and JAX metrics for one (problem, target, model) combo.

    Parameters
    ----------
    pytorch_metrics_path, jax_metrics_path:
        Paths to per-run metrics CSVs (as produced by
        scripts/run_pytorch_sweep.py / scripts/run_jax_sweep.py) for the
        *same* experiment/architecture, one per framework.

    Returns
    -------
    pd.DataFrame
        One row per ``m_train``, with columns:
        ``pytorch_error_gmean``, ``pytorch_error_gstd``,
        ``jax_error_gmean``, ``jax_error_gstd``,
        ``jax_to_pytorch_error_ratio``,
        ``pytorch_time_gmean``, ``jax_time_gmean``,
        ``jax_to_pytorch_time_ratio``.
        A ratio > 1 means JAX is larger (worse for error, slower for time).
    """
    pytorch_frame = _load_and_validate(pytorch_metrics_path)
    jax_frame = _load_and_validate(jax_metrics_path)

    pytorch_error = aggregate_geometric(pytorch_frame, ["m_train"], "relative_test_error")
    pytorch_error = pytorch_error.rename(
        columns={"geometric_mean": "pytorch_error_gmean", "geometric_std": "pytorch_error_gstd"}
    )[["m_train", "pytorch_error_gmean", "pytorch_error_gstd"]]

    jax_error = aggregate_geometric(jax_frame, ["m_train"], "relative_test_error")
    jax_error = jax_error.rename(
        columns={"geometric_mean": "jax_error_gmean", "geometric_std": "jax_error_gstd"}
    )[["m_train", "jax_error_gmean", "jax_error_gstd"]]

    pytorch_time = aggregate_geometric(pytorch_frame, ["m_train"], "training_time_sec")
    pytorch_time = pytorch_time.rename(columns={"geometric_mean": "pytorch_time_gmean"})[
        ["m_train", "pytorch_time_gmean"]
    ]

    jax_time = aggregate_geometric(jax_frame, ["m_train"], "training_time_sec")
    jax_time = jax_time.rename(columns={"geometric_mean": "jax_time_gmean"})[["m_train", "jax_time_gmean"]]

    merged = (
        pytorch_error.merge(jax_error, on="m_train", how="outer")
        .merge(pytorch_time, on="m_train", how="outer")
        .merge(jax_time, on="m_train", how="outer")
        .sort_values("m_train")
        .reset_index(drop=True)
    )

    merged["jax_to_pytorch_error_ratio"] = merged["jax_error_gmean"] / merged["pytorch_error_gmean"]
    merged["jax_to_pytorch_time_ratio"] = merged["jax_time_gmean"] / merged["pytorch_time_gmean"]

    return merged


def compare_convergence_slopes(
    pytorch_metrics_path: str | Path,
    jax_metrics_path: str | Path,
) -> dict[str, dict[str, float]]:
    """Fit and compare each framework's log-log convergence slope.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{"pytorch": {...}, "jax": {...}}``, each value the dict returned
        by ``estimate_slope_from_metrics`` (slope/intercept/num_points/
        min_m/max_m).
    """
    return {
        "pytorch": estimate_slope_from_metrics(pytorch_metrics_path),
        "jax": estimate_slope_from_metrics(jax_metrics_path),
    }


def _load_and_validate(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")

    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError(f"Metrics file is empty: {path}")

    return frame
