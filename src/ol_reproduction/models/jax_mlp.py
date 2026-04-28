"""Pure JAX MLP model implementation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


Array = jax.Array
Params = list[dict[str, Array]]


@dataclass(frozen=True)
class JaxMlpConfig:
    """Configuration for a fully connected JAX MLP."""

    input_dim: int
    output_dim: int
    depth: int
    width: int
    activation: str


def initialize_jax_mlp(
    config: JaxMlpConfig,
    key: Array,
) -> Params:
    """Initialize parameters for a fully connected MLP.

    Parameters
    ----------
    config:
        MLP configuration.
    key:
        JAX PRNG key.

    Returns
    -------
    list[dict[str, jax.Array]]
        List of layer parameter dictionaries.
    """
    _validate_mlp_config(config)

    layer_sizes = (
        [config.input_dim]
        + [config.width for _ in range(config.depth)]
        + [config.output_dim]
    )

    keys = jax.random.split(key, num=len(layer_sizes) - 1)
    params = []

    for layer_index, layer_key in enumerate(keys):
        fan_in = layer_sizes[layer_index]
        fan_out = layer_sizes[layer_index + 1]

        weights_key, _ = jax.random.split(layer_key)

        limit = jnp.sqrt(6.0 / fan_in)
        weights = jax.random.uniform(
            weights_key,
            shape=(fan_in, fan_out),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        )
        biases = jnp.zeros((fan_out,), dtype=jnp.float32)

        params.append(
            {
                "weights": weights,
                "biases": biases,
            }
        )

    return params


def apply_jax_mlp(
    params: Params,
    inputs: Array,
    activation: str,
) -> Array:
    """Apply a JAX MLP to inputs.

    Parameters
    ----------
    params:
        Model parameters.
    inputs:
        Input array with shape ``(batch_size, input_dim)``.
    activation:
        Activation function name.

    Returns
    -------
    jax.Array
        Output array with shape ``(batch_size, output_dim)``.
    """
    hidden = inputs

    for layer in params[:-1]:
        hidden = hidden @ layer["weights"] + layer["biases"]
        hidden = _apply_activation(hidden, activation)

    output_layer = params[-1]
    return hidden @ output_layer["weights"] + output_layer["biases"]


def count_jax_parameters(params: Params) -> int:
    """Count the number of trainable parameters.

    Parameters
    ----------
    params:
        Model parameters.

    Returns
    -------
    int
        Total number of scalar parameters.
    """
    total = 0

    for layer in params:
        total += int(layer["weights"].size)
        total += int(layer["biases"].size)

    return total


def _apply_activation(inputs: Array, activation: str) -> Array:
    """Apply an activation function."""
    normalized_activation = activation.lower()

    if normalized_activation == "relu":
        return jax.nn.relu(inputs)

    if normalized_activation == "elu":
        return jax.nn.elu(inputs)

    if normalized_activation == "tanh":
        return jnp.tanh(inputs)

    raise ValueError(f"Unsupported activation: {activation}")


def _validate_mlp_config(config: JaxMlpConfig) -> None:
    """Validate JAX MLP configuration."""
    if config.input_dim <= 0:
        raise ValueError("input_dim must be positive.")

    if config.output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    if config.depth <= 0:
        raise ValueError("depth must be positive.")

    if config.width <= 0:
        raise ValueError("width must be positive.")

    if config.activation not in {"relu", "elu", "tanh"}:
        raise ValueError(
            "activation must be one of: relu, elu, tanh."
        )