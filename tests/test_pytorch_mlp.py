"""Tests for the PyTorch MLP."""

from __future__ import annotations

import torch

from ol_reproduction.models.pytorch_mlp import PyTorchMLP


def test_pytorch_mlp_output_shape() -> None:
    """MLP output should have shape (batch_size, output_dim)."""
    model = PyTorchMLP(
        input_dim=4,
        output_dim=256,
        depth=4,
        width=40,
        activation="elu",
    )

    inputs = torch.zeros((5, 4))
    outputs = model(inputs)

    assert outputs.shape == (5, 256)


def test_pytorch_mlp_rejects_invalid_activation() -> None:
    """MLP should reject unsupported activation functions."""
    try:
        PyTorchMLP(
            input_dim=4,
            output_dim=256,
            depth=4,
            width=40,
            activation="invalid",
        )
    except ValueError as error:
        assert "Unsupported activation" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")
