"""Tests for the JAX MLP implementation."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

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


@pytest.mark.parametrize("activation", ["relu", "elu", "tanh"])
def test_kaiming_uniform_matches_keras_he_uniform_bound(activation: str) -> None:
    """Keras HeUniform always uses limit = sqrt(6 / fan_in), regardless of
    activation -- verify every supported activation gets that same bound."""
    fan_in = 4
    config = JaxMlpConfig(
        input_dim=fan_in,
        output_dim=40,
        depth=1,
        width=40,
        activation=activation,
        initialization="kaiming_uniform",
    )
    key = jax.random.PRNGKey(0)
    params = initialize_jax_mlp(config=config, key=key)

    first_layer_weights = params[0]["weights"]
    expected_bound = math.sqrt(6.0 / fan_in)

    max_abs_weight = float(jnp.max(jnp.abs(first_layer_weights)))
    assert max_abs_weight <= expected_bound + 1e-6
    assert max_abs_weight > expected_bound * 0.3


def test_default_initialization_matches_kaiming_uniform() -> None:
    """JAX's "default" should match the paper's HeUniform (kaiming_uniform),
    not Xavier -- both should produce the same bound for a given key."""
    config_default = JaxMlpConfig(
        input_dim=4, output_dim=8, depth=1, width=8, activation="elu", initialization="default"
    )
    config_kaiming = JaxMlpConfig(
        input_dim=4, output_dim=8, depth=1, width=8, activation="elu", initialization="kaiming_uniform"
    )

    params_default = initialize_jax_mlp(config=config_default, key=jax.random.PRNGKey(0))
    params_kaiming = initialize_jax_mlp(config=config_kaiming, key=jax.random.PRNGKey(0))

    for layer_default, layer_kaiming in zip(params_default, params_kaiming):
        assert jnp.array_equal(layer_default["weights"], layer_kaiming["weights"])


def test_trial_seed_gives_distinct_but_reproducible_init() -> None:
    config = JaxMlpConfig(
        input_dim=4, output_dim=8, depth=2, width=16, activation="elu", initialization="kaiming_uniform"
    )

    params_seed0_a = initialize_jax_mlp(config=config, key=jax.random.PRNGKey(0))
    params_seed0_b = initialize_jax_mlp(config=config, key=jax.random.PRNGKey(0))
    params_seed1 = initialize_jax_mlp(config=config, key=jax.random.PRNGKey(1))

    assert jnp.array_equal(params_seed0_a[0]["weights"], params_seed0_b[0]["weights"])
    assert not jnp.array_equal(params_seed0_a[0]["weights"], params_seed1[0]["weights"])