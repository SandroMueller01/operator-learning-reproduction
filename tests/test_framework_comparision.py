"""Tests for the PyTorch vs JAX framework comparison (Phase 12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ol_reproduction.evaluation.framework_comparision import (
    compare_convergence_slopes,
    compare_frameworks,
)


def _write_metrics(path, errors_by_m, times_by_m) -> None:
    rows = []
    for m, errors in errors_by_m.items():
        for i, error in enumerate(errors):
            rows.append(
                {
                    "m_train": m,
                    "relative_test_error": error,
                    "training_time_sec": times_by_m[m][i],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_compare_frameworks_ratio_is_one_for_identical_metrics(tmp_path) -> None:
    errors = {10: [1.0, 1.2, 0.8], 100: [0.1, 0.12, 0.08]}
    times = {10: [1.0, 1.0, 1.0], 100: [10.0, 10.0, 10.0]}

    pytorch_path = tmp_path / "pytorch.csv"
    jax_path = tmp_path / "jax.csv"
    _write_metrics(pytorch_path, errors, times)
    _write_metrics(jax_path, errors, times)

    result = compare_frameworks(pytorch_path, jax_path)

    assert np.allclose(result["jax_to_pytorch_error_ratio"].to_numpy(), 1.0, rtol=1e-6)
    assert np.allclose(result["jax_to_pytorch_time_ratio"].to_numpy(), 1.0, rtol=1e-6)


def test_compare_frameworks_detects_jax_being_slower_and_less_accurate(tmp_path) -> None:
    pytorch_errors = {10: [1.0, 1.0]}
    jax_errors = {10: [2.0, 2.0]}  # JAX has 2x the error
    pytorch_times = {10: [1.0, 1.0]}
    jax_times = {10: [3.0, 3.0]}  # JAX takes 3x as long

    pytorch_path = tmp_path / "pytorch.csv"
    jax_path = tmp_path / "jax.csv"
    _write_metrics(pytorch_path, pytorch_errors, pytorch_times)
    _write_metrics(jax_path, jax_errors, jax_times)

    result = compare_frameworks(pytorch_path, jax_path)

    row = result[result["m_train"] == 10].iloc[0]
    assert row["jax_to_pytorch_error_ratio"] == pytest.approx(2.0, rel=1e-6)
    assert row["jax_to_pytorch_time_ratio"] == pytest.approx(3.0, rel=1e-6)


def test_compare_frameworks_rejects_missing_file(tmp_path) -> None:
    jax_path = tmp_path / "jax.csv"
    _write_metrics(jax_path, {10: [1.0]}, {10: [1.0]})

    with pytest.raises(FileNotFoundError):
        compare_frameworks(tmp_path / "missing.csv", jax_path)


def test_compare_convergence_slopes_returns_both_frameworks(tmp_path) -> None:
    # errors ~ 1/m, a clean m^-1 rate for both frameworks.
    errors = {10: [0.1], 100: [0.01], 1000: [0.001]}
    times = {10: [1.0], 100: [1.0], 1000: [1.0]}

    pytorch_path = tmp_path / "pytorch.csv"
    jax_path = tmp_path / "jax.csv"
    _write_metrics(pytorch_path, errors, times)
    _write_metrics(jax_path, errors, times)

    result = compare_convergence_slopes(pytorch_path, jax_path)

    assert set(result.keys()) == {"pytorch", "jax"}
    assert result["pytorch"]["slope"] == pytest.approx(-1.0, rel=1e-6)
    assert result["jax"]["slope"] == pytest.approx(-1.0, rel=1e-6)
