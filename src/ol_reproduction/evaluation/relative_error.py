"""Relative error metrics."""

from __future__ import annotations

import numpy as np


def relative_l2_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
    eps: float = 1.0e-12,
) -> float:
    """Compute relative L2 error over a dataset.

    Parameters
    ----------
    y_true:
        Reference outputs with shape ``(num_samples, output_dim)``.
    y_pred:
        Predicted outputs with shape ``(num_samples, output_dim)``.
    weights:
        Optional sample weights with shape ``(num_samples,)``.
    eps:
        Small number used to avoid division by zero.

    Returns
    -------
    float
        Relative L2 error.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape, "
            f"got {y_true.shape} and {y_pred.shape}."
        )

    difference = y_true - y_pred

    if weights is None:
        numerator = np.sum(difference**2)
        denominator = np.sum(y_true**2)
    else:
        weights = np.asarray(weights)

        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional.")

        if weights.shape[0] != y_true.shape[0]:
            raise ValueError(
                "weights must have length equal to number of samples."
            )

        reshaped_weights = weights.reshape(-1, 1)
        numerator = np.sum(reshaped_weights * difference**2)
        denominator = np.sum(reshaped_weights * y_true**2)

    return float(np.sqrt(numerator / (denominator + eps)))