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
    initialization: str = "default"


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
    params: Params = []

    for layer_index, layer_key in enumerate(keys):
        fan_in = layer_sizes[layer_index]
        fan_out = layer_sizes[layer_index + 1]

        weights_key, _ = jax.random.split(layer_key)

        weights = _initialize_weights(
            key=weights_key,
            fan_in=fan_in,
            fan_out=fan_out,
            initialization=config.initialization,
            activation=config.activation,
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


def _initialize_weights(
    key: Array,
    fan_in: int,
    fan_out: int,
    initialization: str,
    activation: str,
) -> Array:
    """Initialize one dense-layer weight matrix.

    Parameters
    ----------
    key:
        JAX PRNG key.
    fan_in:
        Input dimension of the layer.
    fan_out:
        Output dimension of the layer.
    initialization:
        Initialization scheme.
    activation:
        Activation function name.

    Returns
    -------
    jax.Array
        Initialized weight matrix.
    """
    normalized_name = initialization.lower()

    if normalized_name == "default":
        return _xavier_uniform(
            key=key,
            fan_in=fan_in,
            fan_out=fan_out,
            activation=activation,
        )

    if normalized_name == "kaiming_uniform":
        return _kaiming_uniform(
            key=key,
            fan_in=fan_in,
            fan_out=fan_out,
            activation=activation,
        )

    if normalized_name == "xavier_uniform":
        return _xavier_uniform(
            key=key,
            fan_in=fan_in,
            fan_out=fan_out,
            activation=activation,
        )

    raise ValueError(f"Unsupported initialization: {initialization}")


def _kaiming_uniform(
    key: Array,
    fan_in: int,
    fan_out: int,
    activation: str,
) -> Array:
    """Kaiming-style uniform initialization.

    Parameters
    ----------
    key:
        JAX PRNG key.
    fan_in:
        Input dimension of the layer.
    fan_out:
        Output dimension of the layer.
    activation:
        Activation function name.

    Returns
    -------
    jax.Array
        Initialized weight matrix.
    """
    gain = _kaiming_gain(activation)
    bound = gain * jnp.sqrt(3.0 / fan_in)

    return jax.random.uniform(
        key,
        shape=(fan_in, fan_out),
        minval=-bound,
        maxval=bound,
        dtype=jnp.float32,
    )


def _xavier_uniform(
    key: Array,
    fan_in: int,
    fan_out: int,
    activation: str,
) -> Array:
    """Xavier-style uniform initialization.

    Parameters
    ----------
    key:
        JAX PRNG key.
    fan_in:
        Input dimension of the layer.
    fan_out:
        Output dimension of the layer.
    activation:
        Activation function name.

    Returns
    -------
    jax.Array
        Initialized weight matrix.
    """
    gain = _xavier_gain(activation)
    bound = gain * jnp.sqrt(6.0 / (fan_in + fan_out))

    return jax.random.uniform(
        key,
        shape=(fan_in, fan_out),
        minval=-bound,
        maxval=bound,
        dtype=jnp.float32,
    )


def _apply_activation(inputs: Array, activation: str) -> Array:
    """Apply an activation function.

    Parameters
    ----------
    inputs:
        Input array.
    activation:
        Activation function name.

    Returns
    -------
    jax.Array
        Activated array.
    """
    normalized_activation = activation.lower()

    if normalized_activation == "relu":
        return jax.nn.relu(inputs)

    if normalized_activation == "elu":
        return jax.nn.elu(inputs)

    if normalized_activation == "tanh":
        return jnp.tanh(inputs)

    raise ValueError(f"Unsupported activation: {activation}")


def _kaiming_gain(activation: str) -> float:
    """Return gain for Kaiming-style initialization.

    Parameters
    ----------
    activation:
        Activation function name.

    Returns
    -------
    float
        Gain value.
    """
    normalized_activation = activation.lower()

    if normalized_activation in {"relu", "elu"}:
        return float(jnp.sqrt(2.0))

    return 1.0


def _xavier_gain(activation: str) -> float:
    """Return gain for Xavier-style initialization.

    Parameters
    ----------
    activation:
        Activation function name.

    Returns
    -------
    float
        Gain value.
    """
    normalized_activation = activation.lower()

    if normalized_activation == "tanh":
        return 5.0 / 3.0

    if normalized_activation == "relu":
        return float(jnp.sqrt(2.0))

    return 1.0


def _validate_mlp_config(config: JaxMlpConfig) -> None:
    """Validate JAX MLP configuration.

    Parameters
    ----------
    config:
        MLP configuration.

    Raises
    ------
    ValueError
        If configuration values are invalid.
    """
    if config.input_dim <= 0:
        raise ValueError("input_dim must be positive.")

    if config.output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    if config.depth <= 0:
        raise ValueError("depth must be positive.")

    if config.width <= 0:
        raise ValueError("width must be positive.")

    if config.activation not in {"relu", "elu", "tanh"}:
        raise ValueError("activation must be one of: relu, elu, tanh.")

    if config.initialization not in {
        "default",
        "kaiming_uniform",
        "xavier_uniform",
    }:
        raise ValueError(
            "initialization must be one of: default, kaiming_uniform, "
            "xavier_uniform."
        )