"""Command-line script for plotting activation function comparisons."""

from __future__ import annotations

import argparse

from ol_reproduction.plotting.plot_activation_comparison import (
    plot_activation_comparison,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot activation comparison from metrics CSV files."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="One or more metrics CSV files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the figure should be saved.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--reference-slope",
        type=float,
        default=-1.0,
        help="Reference log-log slope. Default: -1.0.",
    )

    return parser.parse_args()


def main() -> None:
    """Create activation comparison plot."""
    args = parse_args()

    plot_activation_comparison(
        metrics_paths=args.metrics,
        output_path=args.output,
        title=args.title,
        reference_slope=args.reference_slope,
    )

    print(f"Saved activation comparison plot to: {args.output}")


if __name__ == "__main__":
    main()