"""Tests for the JAX MLP implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ol_reproduction.models.jax_mlp import (
    JaxMlpConfig,
    apply_jax_mlp,
    count_jax_parameters,
    initialize_jax_mlp,
)


def test_jax_mlp_output_shape() -> None:
    """JAX MLP output should have shape (batch_size, output_dim)."""
    config = JaxMlpConfig(
        input_dim=4,
        output_dim=256,
        depth=4,
        width=40,
        activation="elu",
        initialization="kaiming_uniform",
    )
    key = jax.random.PRNGKey(0)
    params = initialize_jax_mlp(config=config, key=key)

    inputs = jnp.zeros((5, 4), dtype=jnp.float32)
    outputs = apply_jax_mlp(
        params=params,
        inputs=inputs,
        activation=config.activation,
    )

    assert outputs.shape == (5, 256)


def test_jax_mlp_parameter_count_positive() -> None:
    """Parameter count should be positive."""
    config = JaxMlpConfig(
        input_dim=4,
        output_dim=256,
        depth=4,
        width=40,
        activation="elu",
        initialization="kaiming_uniform",
    )
    key = jax.random.PRNGKey(0)
    params = initialize_jax_mlp(config=config, key=key)

    assert count_jax_parameters(params) > 0


def test_jax_mlp_accepts_xavier_uniform() -> None:
    """JAX MLP should accept Xavier initialization."""
    config = JaxMlpConfig(
        input_dim=4,
        output_dim=256,
        depth=4,
        width=40,
        activation="tanh",
        initialization="xavier_uniform",
    )
    key = jax.random.PRNGKey(0)

    params = initialize_jax_mlp(config=config, key=key)

    assert count_jax_parameters(params) > 0


def test_jax_mlp_rejects_invalid_activation() -> None:
    """JAX MLP should reject unsupported activations."""
    config = JaxMlpConfig(
        input_dim=4,
        output_dim=256,
        depth=4,
        width=40,
        activation="invalid",
        initialization="kaiming_uniform",
    )
    key = jax.random.PRNGKey(0)

    try:
        initialize_jax_mlp(config=config, key=key)
    except ValueError as error:
        assert "activation must be one of" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")


def test_jax_mlp_rejects_invalid_initialization() -> None:
    """JAX MLP should reject unsupported initialization names."""
    config = JaxMlpConfig(
        input_dim=4,
        output_dim=256,
        depth=4,
        width=40,
        activation="elu",
        initialization="invalid",
    )
    key = jax.random.PRNGKey(0)

    try:
        initialize_jax_mlp(config=config, key=key)
    except ValueError as error:
        assert "initialization must be one of" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")