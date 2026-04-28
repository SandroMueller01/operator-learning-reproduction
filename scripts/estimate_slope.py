"""Command-line script for estimating empirical convergence slopes."""

from __future__ import annotations

import argparse
import pprint

from ol_reproduction.evaluation.slope_estimation import (
    estimate_slope_from_metrics,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Estimate log-log convergence slope from metrics CSV."
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics CSV file.",
    )

    return parser.parse_args()


def main() -> None:
    """Estimate and print convergence slope."""
    args = parse_args()

    summary = estimate_slope_from_metrics(metrics_path=args.metrics)

    print("Estimated log-log convergence slope:")
    pprint.pp(summary, sort_dicts=False)


if __name__ == "__main__":
    main()