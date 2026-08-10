"""Tests for dataset input/output utilities."""

from __future__ import annotations

import numpy as np

from ol_reproduction.data.dataset_io import (
    load_metadata,
    load_npz_dataset,
    save_metadata,
    save_npz_dataset,
)


def test_save_and_load_npz_dataset(tmp_path) -> None:
    """Saved NPZ datasets should load with the same array values."""
    path = tmp_path / "dataset.npz"
    arrays = {
        "x": np.ones((3, 4), dtype=np.float32),
        "y_u": np.zeros((3, 16), dtype=np.float32),
    }

    save_npz_dataset(path=path, arrays=arrays)
    loaded = load_npz_dataset(path)

    assert set(loaded.keys()) == {"x", "y_u"}
    np.testing.assert_allclose(loaded["x"], arrays["x"])
    np.testing.assert_allclose(loaded["y_u"], arrays["y_u"])


def test_save_and_load_metadata(tmp_path) -> None:
    """Saved metadata should load with the same content."""
    path = tmp_path / "metadata.json"
    metadata = {
        "experiment": {
            "name": "diffusion_affine_d4",
            "problem": "diffusion",
        },
        "output": {
            "output_dimension": 256,
        },
    }

    save_metadata(path=path, metadata=metadata)
    loaded = load_metadata(path)

    assert loaded == metadata