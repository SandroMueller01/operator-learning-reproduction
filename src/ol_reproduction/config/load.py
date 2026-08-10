"""Utilities for loading configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ConfigDict = dict[str, Any]


def load_yaml(path: str | Path) -> ConfigDict:
    """Load a YAML file and return it as a dictionary.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the given config path does not exist.
    ValueError
        If the YAML file is empty or does not contain a top-level mapping.
    yaml.YAMLError
        If the YAML file contains invalid YAML syntax.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a top-level mapping: {config_path}"
        )

    return config


def load_experiment_config(
    pde_path: str | Path,
    model_path: str | Path,
    train_path: str | Path,
) -> ConfigDict:
    """Load and merge a flat PDE + model + training experiment config.

    Unlike :func:`ol_reproduction.config.resolve.resolve_experiment_case`,
    which selects one case out of a grouped multi-case config, this loads
    three standalone per-concern YAML files and merges their top-level
    sections into a single config dict.

    Parameters
    ----------
    pde_path:
        Path to a flat, single-case PDE config YAML file (top-level sections
        such as ``experiment``, ``pde``, ``coefficient``, ``data``).
    model_path:
        Path to a model architecture config YAML file (top-level sections
        ``model`` and ``initialization``).
    train_path:
        Path to a training config YAML file (top-level section ``training``).

    Returns
    -------
    dict[str, Any]
        Merged experiment config.

    Raises
    ------
    ValueError
        If the same top-level key appears in more than one of the three
        config files.
    """
    pde_config = load_yaml(pde_path)
    model_config = load_yaml(model_path)
    train_config = load_yaml(train_path)

    merged: ConfigDict = {}

    for source_path, partial_config in (
        (pde_path, pde_config),
        (model_path, model_config),
        (train_path, train_config),
    ):
        for key, value in partial_config.items():
            if key in merged:
                raise ValueError(
                    f"Config key {key!r} from {source_path} is already "
                    "defined by another config file. PDE, model, and "
                    "training configs must use disjoint top-level keys."
                )
            merged[key] = value

    return merged