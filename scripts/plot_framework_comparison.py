"""Command-line script for plotting PyTorch/JAX framework comparisons."""

from __future__ import annotations

import argparse

from ol_reproduction.plotting.plot_framework_comparison import (
    plot_framework_comparison,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot PyTorch and JAX framework comparison figures."
    )
    parser.add_argument(
        "--pytorch-metrics",
        required=True,
        help="Path to PyTorch metrics CSV.",
    )
    parser.add_argument(
        "--jax-metrics",
        required=True,
        help="Path to JAX metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where figures should be saved.",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment name used in figure titles and file names.",
    )

    return parser.parse_args()


def main() -> None:
    """Create framework comparison plots."""
    args = parse_args()

    plot_framework_comparison(
        pytorch_metrics_path=args.pytorch_metrics,
        jax_metrics_path=args.jax_metrics,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    print(f"Saved framework comparison plots to: {args.output_dir}")


if __name__ == "__main__":
    main()