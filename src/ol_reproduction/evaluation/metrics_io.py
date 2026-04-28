"""Utilities for saving experiment metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


MetricRow = dict[str, Any]


def append_metrics_row(
    path: str | Path,
    row: MetricRow,
) -> None:
    """Append one metrics row to a CSV file.

    Parameters
    ----------
    path:
        Output CSV path.
    row:
        Metrics row.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def read_metrics_csv(path: str | Path) -> list[MetricRow]:
    """Read metrics rows from a CSV file.

    Parameters
    ----------
    path:
        CSV file path.

    Returns
    -------
    list[dict[str, Any]]
        Metrics rows.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)