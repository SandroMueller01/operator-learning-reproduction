"""Log-transformed parametric coefficient from the target paper."""

from __future__ import annotations

import numpy as np

BETA_C = 0.125
BETA_P = max(1.0, 2.0 * BETA_C)
BETA = BETA_C / BETA_P


def log_transformed_coefficient(
    z1: np.ndarray,
    parameters: np.ndarray,
    base_value: float = 2.62,
) -> np.ndarray:
    """Evaluate the log-transformed diffusion coefficient a_{2,d} (eq. B.2).

    The coefficient is

    .. math::

        a_{2,d}(z, x)
        =
        \\exp\\left(
            1
            + x_1 \\left(\\frac{\\sqrt{\\pi \\beta}}{2}\\right)^{1/2}
            + \\sum_{j=2}^{d} \\zeta_j \\theta_j(z) x_j
        \\right),

    where

    .. math::

        \\zeta_j = (\\sqrt{\\pi \\beta})^{1/2}
            \\exp\\left(-\\frac{(\\lfloor j/2 \\rfloor \\pi \\beta)^2}{8}\\right),

        \\theta_j(z) =
        \\begin{cases}
            \\sin(\\lfloor j/2 \\rfloor \\pi z_1 / \\beta_p) & j \\text{ even} \\\\
            \\cos(\\lfloor j/2 \\rfloor \\pi z_1 / \\beta_p) & j \\text{ odd}
        \\end{cases},

    and :math:`\\beta_c = 1/8`, :math:`\\beta_p = \\max(1, 2 \\beta_c)`,
    :math:`\\beta = \\beta_c / \\beta_p`. This is the paper's actual
    log-transformed coefficient (a rescaled layered coefficient from [76]),
    not simply the exponential of the affine coefficient's sine expansion.

    Parameters
    ----------
    z1:
        First spatial coordinate. Can have any shape, for example
        ``(ny, nx)``.
    parameters:
        Parameter vector of shape ``(d,)``.
    base_value:
        Unused. The paper's a_{2,d} formula has a fixed additive constant
        of 1, not a configurable base value like the affine coefficient's
        2.62. Kept only so callers can pass the same keyword arguments to
        both coefficient functions.

    Returns
    -------
    np.ndarray
        Positive coefficient field with the same shape as ``z1``.

    Raises
    ------
    ValueError
        If ``parameters`` is not one-dimensional.
    """
    del base_value

    parameters = np.asarray(parameters)

    if parameters.ndim != 1:
        raise ValueError(
            "parameters must be one-dimensional with shape (d,), "
            f"got shape {parameters.shape}."
        )

    log_coefficient = np.full_like(z1, fill_value=1.0, dtype=np.float64)

    if parameters.size >= 1:
        log_coefficient = log_coefficient + parameters[0] * np.sqrt(
            np.sqrt(np.pi * BETA) / 2.0
        )

    for j, parameter in enumerate(parameters[1:], start=2):
        half_j = j // 2
        zeta_j = np.sqrt(np.sqrt(np.pi * BETA)) * np.exp(
            -((half_j * np.pi * BETA) ** 2) / 8.0
        )

        if j % 2 == 0:
            theta_j = np.sin(half_j * np.pi * z1 / BETA_P)
        else:
            theta_j = np.cos(half_j * np.pi * z1 / BETA_P)

        log_coefficient = log_coefficient + parameter * zeta_j * theta_j

    return np.exp(log_coefficient)
