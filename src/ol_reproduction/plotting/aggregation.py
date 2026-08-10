"""Geometric mean/std aggregation, matching the paper's reported statistic.

configs/diffusion.yaml and configs/navier_stokes_brinkman.yaml both
document ``evaluation.aggregation: {mean: geometric, spread: geometric_std}``
as the paper's protocol (Fig. 1-3 captions: "geometric mean ... and plus/minus
one (geometric) standard deviation"), since errors are plotted on a
logarithmic y-axis. This module implements that aggregation once, shared by
every plotting script, instead of each one computing (or mis-computing, as
arithmetic mean/std) its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def geometric_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Geometric mean and geometric standard deviation of positive values.

    The geometric mean is ``exp(mean(log(values)))``; the geometric
    standard deviation is the multiplicative factor
    ``exp(std(log(values)))``, so the "plus/minus one geometric std" band
    is ``[gmean / gstd, gmean * gstd]`` (not ``gmean +/- gstd``, which
    would not respect the values' strictly-positive, log-scale nature).

    Parameters
    ----------
    values:
        One-dimensional array of strictly positive values.

    Returns
    -------
    tuple[float, float]
        ``(geometric_mean, geometric_std)``. ``geometric_std`` is ``1.0``
        (a null-width band) if there is only one value.

    Raises
    ------
    ValueError
        If ``values`` is empty or contains a non-positive entry.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        raise ValueError("values must not be empty.")

    if np.any(values <= 0.0):
        raise ValueError("geometric_mean_std requires strictly positive values.")

    log_values = np.log(values)
    gmean = float(np.exp(np.mean(log_values)))
    gstd = float(np.exp(np.std(log_values))) if values.size > 1 else 1.0

    return gmean, gstd


def aggregate_geometric(
    data_frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    """Group a dataframe and compute geometric mean/std per group.

    Parameters
    ----------
    data_frame:
        Input dataframe.
    group_columns:
        Columns to group by (e.g. ``["m_train"]`` or
        ``["framework", "m_train"]``).
    value_column:
        Column to aggregate (e.g. ``"relative_test_error"``).

    Returns
    -------
    pd.DataFrame
        One row per group, with columns ``group_columns + [
        "geometric_mean", "geometric_std", "band_lower", "band_upper"]``,
        sorted by ``group_columns``.
    """
    records = []

    for group_key, group_frame in data_frame.groupby(group_columns):
        gmean, gstd = geometric_mean_std(group_frame[value_column].to_numpy())

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        record = dict(zip(group_columns, group_key))
        record["geometric_mean"] = gmean
        record["geometric_std"] = gstd
        record["band_lower"] = gmean / gstd
        record["band_upper"] = gmean * gstd
        records.append(record)

    result = pd.DataFrame.from_records(records)
    return result.sort_values(group_columns).reset_index(drop=True)
