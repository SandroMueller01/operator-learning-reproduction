"""Generic data-generation dispatcher."""

from __future__ import annotations

from typing import Any

from ol_reproduction.data.generate_diffusion import generate_diffusion_dataset
from ol_reproduction.data.generate_nsb import generate_nsb_dataset

ConfigDict = dict[str, Any]


_DATA_GENERATORS = {
    "diffusion": generate_diffusion_dataset,
    "navier_stokes_brinkman": generate_nsb_dataset,
}


def generate_data_from_config(config: ConfigDict) -> None:
    """Generate data for one resolved experiment config."""
    problem = str(config["experiment"]["problem"])

    if problem not in _DATA_GENERATORS:
        available_problems = sorted(_DATA_GENERATORS)
        raise ValueError(
            f"Unsupported problem {problem!r}. Available problems are: "
            f"{available_problems}."
        )

    generator = _DATA_GENERATORS[problem]
    generator(config)