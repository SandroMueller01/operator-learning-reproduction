"""Tests for the paper Fig. 1/2/3-style multi-panel plot."""

from __future__ import annotations

import pandas as pd
import pytest

from ol_reproduction.plotting.plot_paper_figure import (
    DEFAULT_SUBPLOT_ORDER,
    _parse_problem_name,
    load_and_tag_metrics,
    plot_paper_figure,
)


def _write_case_metrics(path, problem: str, model: str) -> None:
    data_frame = pd.DataFrame(
        {
            "problem": [problem] * 4,
            "model": [model] * 4,
            "m_train": [10, 10, 100, 100],
            "relative_test_error": [1.0, 1.2, 0.1, 0.12],
        }
    )
    data_frame.to_csv(path, index=False)


def test_parse_problem_name() -> None:
    assert _parse_problem_name("diffusion_affine_d4") == ("affine", 4)
    assert _parse_problem_name("nsb_log_d8") == ("log", 8)


def test_parse_problem_name_rejects_unexpected_format() -> None:
    with pytest.raises(ValueError):
        _parse_problem_name("diffusion_affine")


def test_load_and_tag_metrics_adds_coefficient_and_dimension(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    _write_case_metrics(path, "diffusion_affine_d4", "mlp_4x40_elu")

    data_frame = load_and_tag_metrics([path])

    assert list(data_frame["coefficient"].unique()) == ["affine"]
    assert list(data_frame["dimension"].unique()) == [4]


def test_load_and_tag_metrics_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_and_tag_metrics([tmp_path / "missing.csv"])


def test_plot_paper_figure_creates_output_with_partial_data(tmp_path) -> None:
    """Only affine_d4 data is available; the other 3 subplots should still
    render (as "no data yet" placeholders) rather than erroring."""
    csv_paths = []
    for model in ["mlp_4x40_relu", "mlp_4x40_elu", "mlp_4x40_tanh"]:
        path = tmp_path / f"diffusion_affine_d4_{model}.csv"
        _write_case_metrics(path, "diffusion_affine_d4", model)
        csv_paths.append(path)

    output_path = tmp_path / "figures" / "diffusion_fig1.png"

    plot_paper_figure(
        metrics_paths=csv_paths,
        output_path=output_path,
        pde_title="Elliptic diffusion equation",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_paper_figure_all_four_subplots(tmp_path) -> None:
    csv_paths = []
    for coefficient, dimension in DEFAULT_SUBPLOT_ORDER:
        problem = f"diffusion_{coefficient}_d{dimension}"
        for model in ["mlp_4x40_relu", "mlp_4x40_elu"]:
            path = tmp_path / f"{problem}_{model}.csv"
            _write_case_metrics(path, problem, model)
            csv_paths.append(path)

    output_path = tmp_path / "figures" / "diffusion_fig1_full.png"

    plot_paper_figure(
        metrics_paths=csv_paths,
        output_path=output_path,
        pde_title="Elliptic diffusion equation",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
