"""Navier-Stokes-Brinkman dataset generation."""

from __future__ import annotations

from typing import Any

from ol_reproduction.data.generate_dataset import (
    generate_nsb_dataset_from_config,
)

ConfigDict = dict[str, Any]


def generate_nsb_dataset(config: ConfigDict) -> None:
    """Generate NSB dataset from a resolved experiment config."""
    generate_nsb_dataset_from_config(config)