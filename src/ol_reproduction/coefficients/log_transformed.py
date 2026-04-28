"""Log-transformed parametric coefficient from the target paper."""

from __future__ import annotations

import numpy as np


def log_transformed_coefficient(
    z1: np.ndarray,
    parameters: np.ndarray,
    base_value: float = 2.62,
) -> np.ndarray:
    """Evaluate the log-transformed diffusion coefficient.

    The coefficient is implemented as

    .. math::

        a_{2,d}(z, x)
        =
        \\exp\\left(
            \\log(c)
            +
            \\sum_{j=1}^{d}
            x_j
            \\frac{\\sin(\\pi z_1 j)}{j^{3/2}}
        \\right),

    where ``c`` is ``base_value``.

    Parameters
    ----------
    z1:
        First spatial coordinate. Can have any shape, for example
        ``(ny, nx)``.
    parameters:
        Parameter vector with shape ``(d,)``.
    base_value:
        Positive baseline value.

    Returns
    -------
    np.ndarray
        Positive coefficient field with the same shape as ``z1``.

    Raises
    ------
    ValueError
        If ``parameters`` is not one-dimensional or ``base_value`` is not
        positive.
    """
    parameters = np.asarray(parameters)

    if parameters.ndim != 1:
        raise ValueError(
            "parameters must be one-dimensional with shape (d,), "
            f"got shape {parameters.shape}."
        )

    if base_value <= 0.0:
        raise ValueError("base_value must be positive.")

    log_coefficient = np.full_like(
        z1,
        fill_value=np.log(base_value),
        dtype=np.float64,
    )

    for index, parameter in enumerate(parameters, start=1):
        log_coefficient += (
            parameter
            * np.sin(np.pi * z1 * index)
            / (index**1.5)
        )

    return np.exp(log_coefficient)