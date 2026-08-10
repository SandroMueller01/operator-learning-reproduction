"""PyTorch MLP model definitions and initialization utilities."""

from __future__ import annotations

import torch
from torch import nn


def get_activation(name: str) -> nn.Module:
    """Return a PyTorch activation module.

    Parameters
    ----------
    name:
        Activation function name.

    Returns
    -------
    nn.Module
        PyTorch activation module.

    Raises
    ------
    ValueError
        If the activation is not supported.
    """
    normalized_name = name.lower()

    if normalized_name == "relu":
        return nn.ReLU()

    if normalized_name == "elu":
        return nn.ELU()

    if normalized_name == "tanh":
        return nn.Tanh()

    raise ValueError(f"Unsupported activation: {name}")


class PyTorchMLP(nn.Module):
    """Fully connected multilayer perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        width: int,
        activation: str,
    ) -> None:
        """Initialize the MLP.

        Parameters
        ----------
        input_dim:
            Input dimension.
        output_dim:
            Output dimension.
        depth:
            Number of hidden layers.
        width:
            Width of each hidden layer.
        activation:
            Activation function name.

        Raises
        ------
        ValueError
            If dimensions, depth, or width are invalid.
        """
        super().__init__()

        self._validate_dimensions(
            input_dim=input_dim,
            output_dim=output_dim,
            depth=depth,
            width=width,
        )

        layers: list[nn.Module] = []
        current_dim = input_dim

        for _ in range(depth):
            layers.append(nn.Linear(current_dim, width))
            layers.append(get_activation(activation))
            current_dim = width

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the network.

        Parameters
        ----------
        inputs:
            Input tensor with shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor with shape ``(batch_size, output_dim)``.
        """
        return self.network(inputs)

    @staticmethod
    def _validate_dimensions(
        input_dim: int,
        output_dim: int,
        depth: int,
        width: int,
    ) -> None:
        """Validate model dimensions.

        Parameters
        ----------
        input_dim:
            Input dimension.
        output_dim:
            Output dimension.
        depth:
            Number of hidden layers.
        width:
            Hidden-layer width.

        Raises
        ------
        ValueError
            If any argument is not positive.
        """
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        if depth <= 0:
            raise ValueError("depth must be positive.")

        if width <= 0:
            raise ValueError("width must be positive.")


def initialize_weights(
    model: nn.Module,
    initialization_name: str,
    activation: str,
    trial_seed: int | None = None,
) -> None:
    """Initialize model weights according to the requested scheme.

    Parameters
    ----------
    model:
        PyTorch model.
    initialization_name:
        Initialization scheme. Supported values are ``default``,
        ``kaiming_uniform``, and ``xavier_uniform``.
    activation:
        Activation function name. Used to choose initialization gain.
    trial_seed:
        If given, seeds PyTorch's global RNG immediately before drawing
        the initial weights, so different trials (paper: 12 per
        architecture/m/coefficient/dimension combo) get distinct, but
        individually reproducible, initializations. Required for the
        paper's per-trial init protocol (Appendix A.2(iii): weights seeded
        by trial number) -- without it, every trial that shares a config
        file would silently reuse whatever global RNG state happened to be
        active, defeating the point of running multiple trials.

    Raises
    ------
    ValueError
        If the initialization scheme is not supported.
    """
    if trial_seed is not None:
        torch.manual_seed(trial_seed)

    normalized_name = initialization_name.lower()

    if normalized_name == "default":
        return

    if normalized_name == "kaiming_uniform":
        _initialize_kaiming_uniform(
            model=model,
            activation=activation,
        )
        return

    if normalized_name == "xavier_uniform":
        _initialize_xavier_uniform(
            model=model,
            activation=activation,
        )
        return

    raise ValueError(f"Unsupported initialization: {initialization_name}")


def initialize_weights_kaiming_uniform(model: nn.Module) -> None:
    """Initialize linear layers with Kaiming uniform initialization.

    This function is kept for backwards compatibility. New code should use
    ``initialize_weights`` instead.

    Parameters
    ----------
    model:
        PyTorch model.
    """
    _initialize_kaiming_uniform(
        model=model,
        activation="relu",
    )


def _initialize_kaiming_uniform(
    model: nn.Module,
    activation: str,
) -> None:
    """Initialize linear layers with Kaiming uniform initialization.

    Parameters
    ----------
    model:
        PyTorch model.
    activation:
        Activation function name.
    """
    nonlinearity = _activation_to_kaiming_nonlinearity(activation)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(
                module.weight,
                nonlinearity=nonlinearity,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _initialize_xavier_uniform(
    model: nn.Module,
    activation: str,
) -> None:
    """Initialize linear layers with Xavier uniform initialization.

    Parameters
    ----------
    model:
        PyTorch model.
    activation:
        Activation function name.
    """
    gain = _activation_to_xavier_gain(activation)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(
                module.weight,
                gain=gain,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _activation_to_kaiming_nonlinearity(activation: str) -> str:
    """Map activation names to PyTorch Kaiming nonlinearity names.

    The paper initializes with Keras's ``HeUniform``, which is *not*
    activation-aware: it always uses the ReLU-derived bound
    ``limit = sqrt(6 / fan_in)`` regardless of which activation the layer
    actually feeds into. PyTorch's ``kaiming_uniform_`` normally computes a
    different gain per nonlinearity (e.g. gain=1 for "linear"/tanh), so to
    match Keras's behavior exactly this deliberately returns "relu" for
    every supported activation, including tanh -- using PyTorch's own
    tanh-specific gain here would silently diverge from what the paper
    actually initializes with.

    Parameters
    ----------
    activation:
        Activation function name.

    Returns
    -------
    str
        Nonlinearity name accepted by ``torch.nn.init.kaiming_uniform_``.
    """
    normalized_activation = activation.lower()

    if normalized_activation in ("relu", "elu", "tanh"):
        return "relu"

    return "relu"


def _activation_to_xavier_gain(activation: str) -> float:
    """Return Xavier initialization gain for an activation.

    Parameters
    ----------
    activation:
        Activation function name.

    Returns
    -------
    float
        Gain value for Xavier initialization.
    """
    normalized_activation = activation.lower()

    if normalized_activation == "relu":
        return nn.init.calculate_gain("relu")

    if normalized_activation == "tanh":
        return nn.init.calculate_gain("tanh")

    if normalized_activation == "elu":
        return 1.0

    return 1.0