"""JAX counterpart of test_pytorch_train_mass_weighted_error.py -- see that
file for the full rationale. Confirms ``train_jax_from_files`` actually
uses the FEM mass-matrix-weighted Bochner-norm test error when a sidecar
mass-matrix file is present, instead of silently falling back to a plain
unweighted relative L2 error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from ol_reproduction.data.dataset_io import save_npz_dataset
from ol_reproduction.pde.mass_matrix import save_mass_matrix_npz
from ol_reproduction.training.jax_train import (
    _load_mass_matrix_for_target,
    train_jax_from_files,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "mlp_4x40_relu.yaml"
TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "jax_fast_debug.yaml"


def _write_tiny_dataset(directory: Path, output_dim: int = 3) -> None:
    rng = np.random.default_rng(0)
    x_train = rng.uniform(-1.0, 1.0, size=(5, 2)).astype(np.float32)
    y_train = rng.normal(size=(5, output_dim)).astype(np.float32)
    x_test = rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
    y_test = rng.normal(size=(4, output_dim)).astype(np.float32)

    save_npz_dataset(directory / "train_m5_seed0.npz", {"x": x_train, "y_u": y_train})
    save_npz_dataset(directory / "test.npz", {"x": x_test, "y_u": y_test})


def test_load_mass_matrix_for_target_prefers_target_specific_file(tmp_path: Path) -> None:
    generic = sp.identity(3, format="csr")
    specific = sp.identity(3, format="csr") * 2.0
    save_mass_matrix_npz(tmp_path / "mass_matrix.npz", generic)
    save_mass_matrix_npz(tmp_path / "mass_matrix_u.npz", specific)

    loaded = _load_mass_matrix_for_target(tmp_path, "u")

    assert loaded is not None
    assert loaded[0, 0] == pytest.approx(2.0)


def test_load_mass_matrix_for_target_returns_none_when_absent(tmp_path: Path) -> None:
    assert _load_mass_matrix_for_target(tmp_path, "u") is None


def test_mass_matrix_presence_changes_reported_test_error(tmp_path: Path) -> None:
    with_matrix_dir = tmp_path / "with_matrix"
    without_matrix_dir = tmp_path / "without_matrix"
    with_matrix_dir.mkdir()
    without_matrix_dir.mkdir()

    _write_tiny_dataset(with_matrix_dir)
    _write_tiny_dataset(without_matrix_dir)

    mass_matrix = sp.csr_matrix(
        np.array(
            [
                [3.0, 0.5, 0.0],
                [0.5, 3.0, 0.5],
                [0.0, 0.5, 3.0],
            ]
        )
    )
    save_mass_matrix_npz(with_matrix_dir / "mass_matrix.npz", mass_matrix)

    common_kwargs = dict(
        train_file="train_m5_seed0.npz",
        test_file="test.npz",
        model_config_path=MODEL_CONFIG,
        train_config_path=TRAIN_CONFIG,
        target="u",
        trial_seed=0,
    )

    result_with = train_jax_from_files(dataset_dir=with_matrix_dir, **common_kwargs)
    result_without = train_jax_from_files(dataset_dir=without_matrix_dir, **common_kwargs)

    assert result_with["relative_test_error"] != pytest.approx(
        result_without["relative_test_error"], rel=1e-6
    )
