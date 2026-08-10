"""Tests for the PyTorch MLP."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from ol_reproduction.models.pytorch_mlp import PyTorchMLP, initialize_weights


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


@pytest.mark.parametrize("activation", ["relu", "elu", "tanh"])
def test_kaiming_uniform_matches_keras_he_uniform_bound(activation: str) -> None:
    """Keras HeUniform always uses limit = sqrt(6 / fan_in), regardless of
    activation -- verify every supported activation gets that same bound
    (not a narrower/wider one from PyTorch's own activation-aware gain)."""
    model = PyTorchMLP(input_dim=4, output_dim=40, depth=1, width=40, activation=activation)
    initialize_weights(model=model, initialization_name="kaiming_uniform", activation=activation)

    first_layer = next(m for m in model.modules() if isinstance(m, nn.Linear))
    fan_in = first_layer.in_features
    expected_bound = math.sqrt(6.0 / fan_in)

    max_abs_weight = first_layer.weight.detach().abs().max().item()
    assert max_abs_weight <= expected_bound + 1e-6
    # Sanity: bound should actually be exercised (not a degenerate ~0 init).
    assert max_abs_weight > expected_bound * 0.5


def test_trial_seed_gives_distinct_but_reproducible_init() -> None:
    def build_and_init(seed: int) -> torch.Tensor:
        model = PyTorchMLP(input_dim=4, output_dim=8, depth=2, width=16, activation="elu")
        initialize_weights(
            model=model, initialization_name="kaiming_uniform", activation="elu", trial_seed=seed
        )
        first_layer = next(m for m in model.modules() if isinstance(m, nn.Linear))
        return first_layer.weight.detach().clone()

    weights_seed0_a = build_and_init(0)
    weights_seed0_b = build_and_init(0)
    weights_seed1 = build_and_init(1)

    torch.testing.assert_close(weights_seed0_a, weights_seed0_b)
    assert not torch.allclose(weights_seed0_a, weights_seed1)
