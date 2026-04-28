"""Command-line script for generating PDE datasets."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ol_reproduction.common.config import load_yaml
from ol_reproduction.data.generate_dataset import (
    generate_boussinesq_dataset_from_config,
    generate_diffusion_dataset_from_config,
    generate_nsb_dataset_from_config,
)

ConfigDict = dict[str, Any]
DatasetGenerator = Callable[[ConfigDict], None]


_IMPLEMENTED_GENERATORS: dict[str, DatasetGenerator] = {
    "diffusion": generate_diffusion_dataset_from_config,
    "navier_stokes_brinkman": generate_nsb_dataset_from_config,
    "boussinesq": generate_boussinesq_dataset_from_config,
}

_PLANNED_BUT_NOT_IMPLEMENTED: set[str] = set()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate datasets for PDE reproduction experiments."
    )
    parser.add_argument(
        "--pde",
        required=True,
        help="Path to PDE config YAML file.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate the requested PDE dataset."""
    args = parse_args()
    config_path = Path(args.pde)

    config = load_yaml(config_path)
    _validate_pde_only_config(config=config, config_path=config_path)

    problem = str(config["experiment"]["problem"])
    experiment_name = str(config["experiment"]["name"])

    if problem in _PLANNED_BUT_NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"Data generation for problem={problem!r} is planned but not yet "
            "implemented."
        )

    if problem not in _IMPLEMENTED_GENERATORS:
        raise ValueError(
            f"Unsupported problem={problem!r}. Implemented generators are: "
            f"{sorted(_IMPLEMENTED_GENERATORS)}."
        )

    generator = _IMPLEMENTED_GENERATORS[problem]
    generator(config)

    print(f"Dataset generated successfully for experiment: {experiment_name}")


def _validate_pde_only_config(
    config: ConfigDict,
    config_path: Path,
) -> None:
    """Validate the minimal structure of a PDE-only config."""
    required_sections = [
        "experiment",
        "domain",
        "grid",
        "pde",
        "coefficient",
        "data",
        "paths",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing top-level section {section!r} in {config_path}."
            )

        if not isinstance(config[section], dict):
            raise ValueError(
                f"Section {section!r} in {config_path} must be a mapping."
            )

    experiment = config["experiment"]

    required_experiment_keys = [
        "name",
        "problem",
        "coefficient",
        "dimension",
    ]

    for key in required_experiment_keys:
        if key not in experiment:
            raise ValueError(f"Missing experiment.{key} in {config_path}.")

    has_target = "target" in experiment
    has_targets = "targets" in experiment

    if not has_target and not has_targets:
        raise ValueError(
            f"Missing experiment.target or experiment.targets in {config_path}."
        )

    if has_target and has_targets:
        raise ValueError(
            f"Use either experiment.target or experiment.targets in "
            f"{config_path}, not both."
        )


if __name__ == "__main__":
    main()