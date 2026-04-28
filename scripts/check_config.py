"""Command-line script for loading and validating experiment configs."""

from __future__ import annotations

import argparse
import pprint

from ol_reproduction.common.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Load and validate experiment configs."
    )
    parser.add_argument(
        "--pde",
        required=True,
        help="Path to the PDE config YAML file.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model config YAML file.",
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to the training config YAML file.",
    )

    return parser.parse_args()


def main() -> None:
    """Load, validate, and print the merged config."""
    args = parse_args()

    config = load_experiment_config(
        pde_path=args.pde,
        model_path=args.model,
        train_path=args.train,
    )

    print("Config loaded successfully.\n")
    pprint.pp(config, sort_dicts=False)


if __name__ == "__main__":
    main()