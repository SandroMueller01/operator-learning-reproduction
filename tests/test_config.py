"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from ol_reproduction.common.config import load_experiment_config


def test_load_experiment_config() -> None:
    """The initial diffusion/PyTorch debug config should load successfully."""
    repo_root = Path(__file__).resolve().parents[1]

    config = load_experiment_config(
        pde_path=repo_root / "configs/pde/diffusion_affine_d4.yaml",
        model_path=repo_root / "configs/model/mlp_4x40_elu.yaml",
        train_path=repo_root / "configs/train/pytorch_fast_debug.yaml",
    )

    assert config["experiment"]["problem"] == "diffusion"
    assert config["experiment"]["coefficient"] == "affine"
    assert config["experiment"]["dimension"] == 4
    assert config["model"]["type"] == "mlp"
    assert config["model"]["activation"] == "elu"
    assert config["training"]["framework"] == "pytorch"
