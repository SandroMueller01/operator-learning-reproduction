"""Summarize experiment metrics from results/metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "problem",
    "target",
    "framework",
    "model",
    "activation",
    "m_train",
    "seed",
    "final_train_loss",
    "relative_test_error",
    "training_time_sec",
}

LEGACY_DEFAULT_COLUMNS = {
    "target": "u",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Summarize all experiment metrics CSV files."
    )
    parser.add_argument(
        "--metrics-dir",
        default="results/metrics",
        help="Directory containing metrics CSV files.",
    )
    parser.add_argument(
        "--output",
        default="results/tables/summary_metrics.csv",
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print the summary table to the terminal.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require all metrics files to contain all columns. "
            "By default, known legacy files are repaired when possible."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Create and save a summary table."""
    args = parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_path = Path(args.output)

    summary = summarize_metrics_directory(
        metrics_dir=metrics_dir,
        strict=args.strict,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"Saved summary table to: {output_path}")

    if args.print_table:
        print(summary.to_string(index=False))


def summarize_metrics_directory(
    metrics_dir: str | Path,
    strict: bool = False,
) -> pd.DataFrame:
    """Summarize all metrics CSV files in a directory.

    Parameters
    ----------
    metrics_dir:
        Directory containing metrics CSV files.
    strict:
        If true, reject files with missing columns. If false, repair known
        legacy metrics files when possible.

    Returns
    -------
    pd.DataFrame
        Summary table.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found.
    """
    metrics_dir = Path(metrics_dir)
    csv_paths = sorted(metrics_dir.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {metrics_dir}")

    frames = []

    for csv_path in csv_paths:
        frame = _load_metrics_file(
            csv_path=csv_path,
            strict=strict,
        )
        frame["source_file"] = csv_path.name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    return _summarize_metrics(combined)


def _load_metrics_file(
    csv_path: Path,
    strict: bool,
) -> pd.DataFrame:
    """Load and validate one metrics CSV file.

    Parameters
    ----------
    csv_path:
        Path to metrics CSV.
    strict:
        If true, do not repair missing legacy columns.

    Returns
    -------
    pd.DataFrame
        Loaded metrics dataframe.

    Raises
    ------
    ValueError
        If the file is empty or missing required columns.
    """
    frame = pd.read_csv(csv_path)

    if frame.empty:
        raise ValueError(f"Metrics file is empty: {csv_path}")

    if not strict:
        frame = _repair_legacy_metrics_frame(
            frame=frame,
            csv_path=csv_path,
        )

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Metrics file {csv_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return frame


def _repair_legacy_metrics_frame(
    frame: pd.DataFrame,
    csv_path: Path,
) -> pd.DataFrame:
    """Repair known legacy metrics files.

    Older metrics files were generated before target-aware training and may not
    contain the ``target`` column. In those cases, the target is inferred from
    the filename when possible, otherwise it defaults to ``u``.

    Parameters
    ----------
    frame:
        Metrics dataframe.
    csv_path:
        Source CSV path.

    Returns
    -------
    pd.DataFrame
        Possibly repaired dataframe.
    """
    repaired = frame.copy()

    for column, default_value in LEGACY_DEFAULT_COLUMNS.items():
        if column not in repaired.columns:
            repaired[column] = _infer_legacy_column_value(
                column=column,
                default_value=default_value,
                csv_path=csv_path,
            )

    return repaired


def _infer_legacy_column_value(
    column: str,
    default_value: str,
    csv_path: Path,
) -> str:
    """Infer a missing legacy column value.

    Parameters
    ----------
    column:
        Missing column name.
    default_value:
        Fallback value.
    csv_path:
        Source CSV path.

    Returns
    -------
    str
        Inferred value.
    """
    if column == "target":
        filename = csv_path.name

        if "_phi_" in filename:
            return "phi"

        if "_p_" in filename:
            return "p"

        if "_u_" in filename:
            return "u"

        return default_value

    return default_value


def _summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics from combined metrics.

    Parameters
    ----------
    frame:
        Combined metrics dataframe.

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    group_columns = [
        "problem",
        "target",
        "framework",
        "model",
        "activation",
    ]

    summary = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            num_runs=("relative_test_error", "count"),
            num_train_sizes=("m_train", "nunique"),
            num_seeds=("seed", "nunique"),
            min_m=("m_train", "min"),
            max_m=("m_train", "max"),
            best_relative_error=("relative_test_error", "min"),
            mean_relative_error=("relative_test_error", "mean"),
            std_relative_error=("relative_test_error", "std"),
            mean_final_train_loss=("final_train_loss", "mean"),
            mean_training_time_sec=("training_time_sec", "mean"),
            total_training_time_sec=("training_time_sec", "sum"),
        )
        .sort_values(
            [
                "problem",
                "target",
                "framework",
                "best_relative_error",
            ]
        )
        .reset_index(drop=True)
    )

    summary["std_relative_error"] = summary["std_relative_error"].fillna(0.0)

    return summary


if __name__ == "__main__":
    main()