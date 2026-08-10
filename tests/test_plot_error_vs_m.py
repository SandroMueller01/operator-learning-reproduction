"""Tests for error-versus-m plotting."""

from __future__ import annotations

import pandas as pd

from ol_reproduction.plotting.plot_error_vs_m import plot_error_vs_m


def test_plot_error_vs_m_creates_output_file(tmp_path) -> None:
    """The plotting function should create an output figure."""
    metrics_path = tmp_path / "metrics.csv"
    output_path = tmp_path / "figure.png"

    data_frame = pd.DataFrame(
        {
            "m_train": [10, 20, 50, 100],
            "relative_test_error": [1.0, 0.5, 0.2, 0.1],
        }
    )
    data_frame.to_csv(metrics_path, index=False)

    plot_error_vs_m(
        metrics_path=metrics_path,
        output_path=output_path,
        title="Test Plot",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0