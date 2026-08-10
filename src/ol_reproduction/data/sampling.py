"""Sampling utilities for parameter vectors."""

from __future__ import annotations

import numpy as np


def sample_uniform_parameters(
    num_samples: int,
    dimension: int,
    seed: int | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Sample parameter vectors uniformly from [-1, 1]^d.

    Parameters
    ----------
    num_samples:
        Number of parameter vectors to sample.
    dimension:
        Parametric dimension d.
    seed:
        Optional random seed.
    dtype:
        NumPy dtype of the returned array.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_samples, dimension)``.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if dimension <= 0:
        raise ValueError("dimension must be positive.")

    rng = np.random.default_rng(seed)
    samples = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(num_samples, dimension),
    )

    return samples.astype(dtype)