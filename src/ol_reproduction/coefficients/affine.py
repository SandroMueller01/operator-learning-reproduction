"""Affine parametric coefficient from the target paper."""

from __future__ import annotations

import numpy as np


def affine_coefficient(
    z1: np.ndarray,
    parameters: np.ndarray,
    base_value: float = 2.62,
) -> np.ndarray:
    """Evaluate the affine diffusion coefficient.

    The coefficient is

    .. math::

        a_{1,d}(z, x)
        =
        2.62
        +
        \\sum_{j=1}^{d}
        x_j
        \\frac{\\sin(\\pi z_1 j)}{j^{3/2}}.

    It depends only on the first spatial coordinate ``z1``.

    Parameters
    ----------
    z1:
        First spatial coordinate. Can have any shape, for example
        ``(ny, nx)``.
    parameters:
        Parameter vector of shape ``(d,)``.
    base_value:
        Constant offset in the coefficient.

    Returns
    -------
    np.ndarray
        Coefficient field with the same shape as ``z1``.

    Raises
    ------
    ValueError
        If ``parameters`` is not one-dimensional.
    """
    parameters = np.asarray(parameters)

    if parameters.ndim != 1:
        raise ValueError(
            "parameters must be one-dimensional with shape (d,), "
            f"got shape {parameters.shape}."
        )

    coefficient = np.full_like(
        z1,
        fill_value=base_value,
        dtype=np.float64,
    )

    for index, parameter in enumerate(parameters, start=1):
        coefficient += (
            parameter
            * np.sin(np.pi * z1 * index)
            / (index**1.5)
        )

    return coefficient