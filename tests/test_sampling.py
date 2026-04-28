"""Tests for parameter sampling utilities."""

from __future__ import annotations

import numpy as np

from ol_reproduction.data.sampling import sample_uniform_parameters


def test_sample_uniform_parameters_shape() -> None:
    """Sampling should return an array with shape (num_samples, dimension)."""
    samples = sample_uniform_parameters(
        num_samples=10,
        dimension=4,
        seed=0,
    )

    assert samples.shape == (10, 4)


def test_sample_uniform_parameters_range() -> None:
    """All sampled values should lie in [-1, 1]."""
    samples = sample_uniform_parameters(
        num_samples=100,
        dimension=8,
        seed=0,
    )

    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)


def test_sample_uniform_parameters_reproducible() -> None:
    """Using the same seed should produce the same samples."""
    samples_a = sample_uniform_parameters(
        num_samples=20,
        dimension=4,
        seed=123,
    )
    samples_b = sample_uniform_parameters(
        num_samples=20,
        dimension=4,
        seed=123,
    )

    np.testing.assert_allclose(samples_a, samples_b)


def test_sample_uniform_parameters_invalid_num_samples() -> None:
    """Sampling with non-positive num_samples should fail."""
    try:
        sample_uniform_parameters(num_samples=0, dimension=4)
    except ValueError as error:
        assert "num_samples must be positive" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")


def test_sample_uniform_parameters_invalid_dimension() -> None:
    """Sampling with non-positive dimension should fail."""
    try:
        sample_uniform_parameters(num_samples=10, dimension=0)
    except ValueError as error:
        assert "dimension must be positive" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")