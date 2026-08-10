"""Command-line script for plotting relative test error versus m."""

from __future__ import annotations

import argparse

from ol_reproduction.plotting.plot_error_vs_m import plot_error_vs_m


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot relative test error versus training size."
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics CSV file.",
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
    """Create the error-versus-m plot."""
    args = parse_args()

    plot_error_vs_m(
        metrics_path=args.metrics,
        output_path=args.output,
        title=args.title,
        reference_slope=args.reference_slope,
    )

    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()