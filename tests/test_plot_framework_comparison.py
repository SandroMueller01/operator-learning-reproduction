"""Tests for the PyTorch/JAX framework comparison plots."""

from __future__ import annotations

import pandas as pd
import pytest

from ol_reproduction.plotting.plot_framework_comparison import (
    plot_framework_comparison,
)


def _write_metrics(path, framework: str) -> None:
    data_frame = pd.DataFrame(
        {
            "framework": [framework] * 6,
            "m_train": [10, 10, 10, 100, 100, 100],
            "relative_test_error": [1.0, 1.2, 0.8, 0.1, 0.12, 0.08],
            "training_time_sec": [0.5, 0.6, 0.4, 5.0, 6.0, 4.0],
        }
    )
    data_frame.to_csv(path, index=False)


def test_plot_framework_comparison_creates_both_figures(tmp_path) -> None:
    pytorch_path = tmp_path / "pytorch_metrics.csv"
    jax_path = tmp_path / "jax_metrics.csv"
    _write_metrics(pytorch_path, "pytorch")
    _write_metrics(jax_path, "jax")

    output_dir = tmp_path / "figures"

    plot_framework_comparison(
        pytorch_metrics_path=pytorch_path,
        jax_metrics_path=jax_path,
        output_dir=output_dir,
        experiment_name="diffusion_affine_d4",
    )

    error_path = output_dir / "diffusion_affine_d4_framework_error.png"
    time_path = output_dir / "diffusion_affine_d4_framework_time.png"
    assert error_path.exists() and error_path.stat().st_size > 0
    assert time_path.exists() and time_path.stat().st_size > 0


def test_plot_framework_comparison_rejects_missing_metrics_file(tmp_path) -> None:
    missing_path = tmp_path / "does_not_exist.csv"
    jax_path = tmp_path / "jax_metrics.csv"
    _write_metrics(jax_path, "jax")

    with pytest.raises(FileNotFoundError):
        plot_framework_comparison(
            pytorch_metrics_path=missing_path,
            jax_metrics_path=jax_path,
            output_dir=tmp_path / "figures",
            experiment_name="diffusion_affine_d4",
        )
