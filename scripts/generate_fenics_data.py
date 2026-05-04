"""Command-line script for generating FEniCS datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ol_reproduction.common.config import load_yaml
from ol_reproduction.data.generate_fenics_dataset import (
    generate_fenics_dataset_from_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate FEniCS datasets for paper-aligned experiments."
    )
    parser.add_argument(
        "--pde",
        required=True,
        help="Path to PDE config YAML file.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate FEniCS dataset."""
    args = parse_args()
    config = load_yaml(Path(args.pde))

    generate_fenics_dataset_from_config(config)


if __name__ == "__main__":
    main()