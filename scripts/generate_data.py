"""Command-line script for generating PDE datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ol_reproduction.config.load import load_yaml
from ol_reproduction.config.resolve import resolve_experiment_case
from ol_reproduction.data.generate import generate_data_from_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate datasets for PDE reproduction experiments."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to grouped experiment config YAML file.",
    )
    parser.add_argument(
        "--case",
        required=True,
        help=(
            "Experiment case to run, for example 'affine_d4', "
            "'log_d4', 'affine_d8', or 'log_d8'."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Generate the requested PDE dataset."""
    args = parse_args()

    grouped_config_path = Path(args.config)

    grouped_config = load_yaml(grouped_config_path)
    resolved_config = resolve_experiment_case(
        config=grouped_config,
        case_name=args.case,
    )

    generate_data_from_config(resolved_config)

    experiment_name = str(resolved_config["experiment"]["name"])
    print(f"Dataset generated successfully for experiment: {experiment_name}")


if __name__ == "__main__":
    main()