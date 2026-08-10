"""Tests for the --resume skip-if-completed logic in the sweep scripts
(scripts/run_pytorch_sweep.py, scripts/run_jax_sweep.py) -- Phase 11 prep,
needed so a multi-day sweep can survive being interrupted across sessions.

scripts/ isn't part of the installed package, so these modules are loaded
directly by file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pytorch_sweep_module():
    return _load_script_module("_test_run_pytorch_sweep", "scripts/run_pytorch_sweep.py")


@pytest.fixture(scope="module")
def jax_sweep_module():
    return _load_script_module("_test_run_jax_sweep", "scripts/run_jax_sweep.py")


def _write_metrics_csv(path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_completed_rows_empty_when_file_missing(pytorch_sweep_module, tmp_path) -> None:
    completed = pytorch_sweep_module._load_completed_rows(str(tmp_path / "missing.csv"))
    assert completed == set()


def test_load_completed_rows_parses_m_and_seed(pytorch_sweep_module, tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    _write_metrics_csv(
        path,
        [
            {"m_train": 10, "seed": 0, "relative_test_error": 0.5},
            {"m_train": 10, "seed": 1, "relative_test_error": 0.4},
            {"m_train": 20, "seed": 0, "relative_test_error": 0.3},
        ],
    )

    completed = pytorch_sweep_module._load_completed_rows(str(path))

    assert completed == {(10, 0), (10, 1), (20, 0)}


def test_jax_sweep_module_has_matching_resume_logic(jax_sweep_module, tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    _write_metrics_csv(path, [{"m_train": 100, "seed": 5, "relative_test_error": 0.1}])

    completed = jax_sweep_module._load_completed_rows(str(path))

    assert completed == {(100, 5)}


def test_default_test_file_template_is_shared_test_npz(pytorch_sweep_module, jax_sweep_module, monkeypatch) -> None:
    """Phase 5/6/7 generate one shared test.npz per case, not a per-seed
    test_seed{seed}.npz -- the sweep scripts' default must match, or every
    training run silently evaluates against the wrong (or a missing) file."""
    required_args = ["--dataset", "d", "--model", "m.yaml", "--train", "t.yaml", "--output", "o.csv"]

    for module in (pytorch_sweep_module, jax_sweep_module):
        monkeypatch.setattr(sys, "argv", ["prog", *required_args])
        args = module.parse_args()
        assert args.test_file_template == "test.npz"
