"""Tests for geometric mean/std aggregation (paper Fig. 1-3 statistic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ol_reproduction.plotting.aggregation import (
    aggregate_geometric,
    geometric_mean_std,
)


def test_geometric_mean_matches_hand_computation() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])

    gmean, gstd = geometric_mean_std(values)

    expected_gmean = np.exp(np.mean(np.log(values)))
    assert gmean == pytest.approx(expected_gmean)
    assert gmean == pytest.approx(2.8284271247, rel=1e-8)  # (1*2*4*8)^(1/4)


def test_geometric_mean_of_constant_values_equals_the_value() -> None:
    gmean, gstd = geometric_mean_std(np.array([3.0, 3.0, 3.0]))

    assert gmean == pytest.approx(3.0)
    assert gstd == pytest.approx(1.0)


def test_single_value_has_unit_geometric_std() -> None:
    gmean, gstd = geometric_mean_std(np.array([5.0]))

    assert gmean == pytest.approx(5.0)
    assert gstd == pytest.approx(1.0)


def test_rejects_empty_array() -> None:
    with pytest.raises(ValueError):
        geometric_mean_std(np.array([]))


def test_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError):
        geometric_mean_std(np.array([1.0, -2.0, 3.0]))

    with pytest.raises(ValueError):
        geometric_mean_std(np.array([1.0, 0.0, 3.0]))


def test_aggregate_geometric_single_group_column() -> None:
    data_frame = pd.DataFrame(
        {
            "m_train": [10, 10, 10, 100, 100, 100],
            "relative_test_error": [1.0, 2.0, 4.0, 0.1, 0.2, 0.4],
        }
    )

    result = aggregate_geometric(data_frame, ["m_train"], "relative_test_error")

    assert list(result["m_train"]) == [10, 100]
    row_10 = result[result["m_train"] == 10].iloc[0]
    assert row_10["geometric_mean"] == pytest.approx(2.0, rel=1e-6)  # (1*2*4)^(1/3)
    row_100 = result[result["m_train"] == 100].iloc[0]
    assert row_100["geometric_mean"] == pytest.approx(0.2, rel=1e-6)

    # band should bracket the geometric mean multiplicatively.
    assert row_10["band_lower"] == pytest.approx(row_10["geometric_mean"] / row_10["geometric_std"])
    assert row_10["band_upper"] == pytest.approx(row_10["geometric_mean"] * row_10["geometric_std"])


def test_aggregate_geometric_multiple_group_columns() -> None:
    data_frame = pd.DataFrame(
        {
            "framework": ["pytorch", "pytorch", "jax", "jax"],
            "m_train": [10, 10, 10, 10],
            "relative_test_error": [1.0, 2.0, 0.5, 0.5],
        }
    )

    result = aggregate_geometric(data_frame, ["framework", "m_train"], "relative_test_error")

    assert set(result["framework"]) == {"pytorch", "jax"}
    jax_row = result[result["framework"] == "jax"].iloc[0]
    assert jax_row["geometric_mean"] == pytest.approx(0.5)
    assert jax_row["geometric_std"] == pytest.approx(1.0)
