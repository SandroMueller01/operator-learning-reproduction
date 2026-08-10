"""Dataset input/output utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        Created directory path.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_npz_dataset(
    path: str | Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Save a dataset as a compressed NPZ file.

    Parameters
    ----------
    path:
        Output file path.
    arrays:
        Dictionary of arrays to save.
    """
    output_path = Path(path)
    ensure_directory(output_path.parent)

    np.savez_compressed(output_path, **arrays)


def load_npz_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load a compressed NPZ dataset.

    Parameters
    ----------
    path:
        Path to the dataset file.

    Returns
    -------
    dict[str, np.ndarray]
        Loaded arrays.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    with np.load(dataset_path) as data:
        return {key: data[key] for key in data.files}


def save_metadata(
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Save metadata as JSON.

    Parameters
    ----------
    path:
        Output JSON path.
    metadata:
        Metadata dictionary.
    """
    output_path = Path(path)
    ensure_directory(output_path.parent)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Load metadata from JSON.

    Parameters
    ----------
    path:
        JSON metadata path.

    Returns
    -------
    dict[str, Any]
        Loaded metadata.
    """
    metadata_path = Path(path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Metadata JSON must contain an object.")

    return metadata