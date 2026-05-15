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

# Common type aliases used to make function signatures easier to read.
ConfigDict = dict[str, Any]
DatasetGenerator = Callable[[ConfigDict], None]


# Dispatch table mapping the problem name from the YAML config to the
# corresponding dataset-generation function.
_implemented_generators: dict[str, DatasetGenerator] = {
    "diffusion": generate_diffusion_dataset_from_config,
    "navier_stokes_brinkman": generate_nsb_dataset_from_config,
    "boussinesq": generate_boussinesq_dataset_from_config,
}

# Placeholder for known PDE problems that are planned but not implemented yet.
# Keeping this separate from unsupported problems gives clearer error messages.
_planned_but_not_implemented: set[str] = set()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate datasets for PDE reproduction experiments."
    )

    # The PDE config defines the full data-generation setup, including the
    # problem type, coefficient, grid, boundary conditions, and dataset sizes.
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

    # Load the YAML file into a Python dictionary and validate its minimum
    # required structure before dispatching to a specific generator.
    config = load_yaml(config_path)
    _validate_pde_only_config(config=config, config_path=config_path)

    # The problem field selects the dataset generator. The experiment name is
    # used only for user-facing output after successful generation.
    problem = str(config["experiment"]["problem"])
    experiment_name = str(config["experiment"]["name"])

    # Distinguish between planned future work and completely unsupported
    # problem names.
    if problem in _planned_but_not_implemented:
        raise NotImplementedError(
            f"Data generation for problem={problem!r} is planned but not yet "
            "implemented."
        )

    # Fail early if the config requests a problem for which no generator exists.
    if problem not in _implemented_generators:
        raise ValueError(
            f"Unsupported problem={problem!r}. Implemented generators are: "
            f"{sorted(_implemented_generators)}."
        )

    # Dispatch to the selected generator. For example, problem="diffusion"
    # calls generate_diffusion_dataset_from_config(config).
    generator = _implemented_generators[problem]
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

    # Each required top-level YAML section must exist and must be a mapping.
    # This catches malformed configs before the numerical generation starts.
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

    # These fields define the identity of the experiment and the input
    # parameterization used for dataset generation.
    required_experiment_keys = [
        "name",
        "problem",
        "coefficient",
        "dimension",
    ]

    for key in required_experiment_keys:
        if key not in experiment:
            raise ValueError(f"Missing experiment.{key} in {config_path}.")

    # A config must define either one target, such as "u", or multiple targets,
    # such as ["u", "p"]. Defining both would make the output ambiguous.
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