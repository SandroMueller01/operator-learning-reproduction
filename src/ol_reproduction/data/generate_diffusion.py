"""Diffusion dataset generation."""

from __future__ import annotations

from typing import Any

from ol_reproduction.data.generate_dataset import (
    generate_diffusion_dataset_from_config,
)

ConfigDict = dict[str, Any]


def generate_diffusion_dataset(config: ConfigDict) -> None:
    """Generate diffusion dataset from a resolved experiment config."""
    generate_diffusion_dataset_from_config(config)