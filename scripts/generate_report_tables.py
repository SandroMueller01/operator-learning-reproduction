"""Generate the headline comparison tables referenced from RESULTS.md and
the practical-work report's experimental-results chapter, directly from
results/metrics/*.csv.

Produces, under results/tables/:
    diffusion_table_m500.csv           -- geometric-mean error at m=500, per
                                           (case, architecture, framework)
    diffusion_table_dimension.csv      -- d=8/d=4 error ratio at m=500, per
                                           (coefficient, architecture)
    diffusion_table_slopes.csv         -- fitted log-log decay slope, per
                                           (case, architecture, framework)
    diffusion_table_framework_ratio.csv -- geometric-mean PyTorch/JAX error
                                           and time ratio, per case
    nsb_table_m500.csv                 -- as above, split additionally by
                                           target (u, p)
    nsb_table_dimension.csv
    nsb_table_slopes.csv
    nsb_table_framework_ratio.csv
    nsb_table_activation.csv           -- geometric-mean error at m=500,
                                           pooled over case/framework/size,
                                           split only by (target, activation)

No reusable script previously produced these tables (they were built ad
hoc during earlier analysis); this script replaces that ad hoc process so
regenerating them after a re-run is a single command.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = REPO_ROOT / "results" / "metrics"
TABLES_DIR = REPO_ROOT / "results" / "tables"

PROBLEM_PATTERN = re.compile(r"^(?P<pde>diffusion|nsb)_(?P<coefficient>affine|log)_d(?P<dimension>\d+)$")

M_LARGE = 500


def geometric_mean_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    log_values = np.log(values)
    gmean = float(np.exp(np.mean(log_values)))
    gstd = float(np.exp(np.std(log_values))) if values.size > 1 else 1.0
    return gmean, gstd


def load_all_metrics() -> pd.DataFrame:
    frames = []
    for path in sorted(METRICS_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    parsed = combined["problem"].apply(_parse_problem)
    combined["pde"] = [p[0] for p in parsed]
    combined["coefficient"] = [p[1] for p in parsed]
    combined["dimension"] = [p[2] for p in parsed]

    if "target" not in combined.columns:
        combined["target"] = "u"
    combined["target"] = combined["target"].fillna("u")

    return combined


def _parse_problem(problem: str) -> tuple[str, str, int]:
    match = PROBLEM_PATTERN.match(str(problem))
    if match is None:
        raise ValueError(f"Unexpected problem name: {problem!r}")
    return match.group("pde"), match.group("coefficient"), int(match.group("dimension"))


def build_m500_table(df: pd.DataFrame, pde: str, with_target: bool) -> pd.DataFrame:
    subset = df[(df["pde"] == pde) & (df["m_train"] == M_LARGE)]
    group_cols = ["problem", "target", "model", "activation", "framework"] if with_target else [
        "problem",
        "model",
        "activation",
        "framework",
    ]

    records = []
    for key, group in subset.groupby(group_cols):
        gmean, gstd = geometric_mean_std(group["relative_test_error"].to_numpy())
        record = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        record["case"] = record.pop("problem")
        record["architecture"] = record.pop("model")
        record["gmean_error_m500"] = gmean
        record["gstd_error_m500"] = gstd
        record["n_trials"] = int(len(group))
        records.append(record)

    columns = (
        ["case", "target", "architecture", "activation", "framework", "gmean_error_m500", "gstd_error_m500", "n_trials"]
        if with_target
        else ["case", "architecture", "activation", "framework", "gmean_error_m500", "gstd_error_m500", "n_trials"]
    )
    result = pd.DataFrame.from_records(records)[columns]
    sort_cols = ["case", "target", "architecture", "framework"] if with_target else ["case", "architecture", "framework"]
    return result.sort_values(sort_cols).reset_index(drop=True)


def build_dimension_table(df: pd.DataFrame, pde: str, with_target: bool) -> pd.DataFrame:
    subset = df[(df["pde"] == pde) & (df["m_train"] == M_LARGE)]
    group_cols = ["coefficient", "target", "model", "dimension"] if with_target else ["coefficient", "model", "dimension"]

    pooled = {}
    for key, group in subset.groupby(group_cols):
        gmean, _ = geometric_mean_std(group["relative_test_error"].to_numpy())
        pooled[key] = gmean

    rows = []
    seen = set()
    for key in pooled:
        base_key = key[:-1]
        if base_key in seen:
            continue
        seen.add(base_key)
        key4 = base_key + (4,)
        key8 = base_key + (8,)
        if key4 not in pooled or key8 not in pooled:
            continue
        row = dict(zip(group_cols[:-1], base_key))
        row["architecture"] = row.pop("model")
        row["gmean_error_d4"] = pooled[key4]
        row["gmean_error_d8"] = pooled[key8]
        row["ratio_d8_over_d4"] = pooled[key8] / pooled[key4]
        rows.append(row)

    columns = (
        ["coefficient", "target", "architecture", "gmean_error_d4", "gmean_error_d8", "ratio_d8_over_d4"]
        if with_target
        else ["coefficient", "architecture", "gmean_error_d4", "gmean_error_d8", "ratio_d8_over_d4"]
    )
    result = pd.DataFrame.from_records(rows)[columns]
    sort_cols = ["coefficient", "target", "architecture"] if with_target else ["coefficient", "architecture"]
    return result.sort_values(sort_cols).reset_index(drop=True)


def build_slopes_table(df: pd.DataFrame, pde: str, with_target: bool) -> pd.DataFrame:
    subset = df[df["pde"] == pde]
    group_cols = ["problem", "target", "model", "framework"] if with_target else ["problem", "model", "framework"]

    rows = []
    for key, group in subset.groupby(group_cols):
        per_m = group.groupby("m_train")["relative_test_error"].apply(
            lambda values: geometric_mean_std(values.to_numpy())[0]
        )
        per_m = per_m.sort_index()
        if len(per_m) < 2:
            continue
        slope, _ = np.polyfit(np.log(per_m.index.to_numpy(dtype=float)), np.log(per_m.to_numpy()), deg=1)

        record = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        record["case"] = record.pop("problem")
        record["architecture"] = record.pop("model")
        record["fitted_slope"] = float(slope)
        rows.append(record)

    columns = (
        ["case", "target", "architecture", "framework", "fitted_slope"]
        if with_target
        else ["case", "architecture", "framework", "fitted_slope"]
    )
    result = pd.DataFrame.from_records(rows)[columns]
    sort_cols = ["case", "target", "architecture", "framework"] if with_target else ["case", "architecture", "framework"]
    return result.sort_values(sort_cols).reset_index(drop=True)


def build_framework_ratio_table(df: pd.DataFrame, pde: str, with_target: bool) -> pd.DataFrame:
    subset = df[df["pde"] == pde]
    group_cols = ["problem", "target"] if with_target else ["problem"]

    rows = []
    for key, group in subset.groupby(group_cols):
        pt = group[group["framework"] == "pytorch"].set_index(["model", "m_train", "seed"])
        jx = group[group["framework"] == "jax"].set_index(["model", "m_train", "seed"])
        matched_index = pt.index.intersection(jx.index)

        error_ratios = (pt.loc[matched_index, "relative_test_error"] / jx.loc[matched_index, "relative_test_error"]).to_numpy()
        time_ratios = (pt.loc[matched_index, "training_time_sec"] / jx.loc[matched_index, "training_time_sec"]).to_numpy()

        error_gmean, _ = geometric_mean_std(error_ratios)
        time_gmean, _ = geometric_mean_std(time_ratios)

        record = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        record["case"] = record.pop("problem")
        record["n_matched_runs"] = int(len(matched_index))
        record["gmean_error_ratio_pt_over_jax"] = error_gmean
        record["gmean_time_ratio_pt_over_jax"] = time_gmean
        rows.append(record)

    columns = (
        ["case", "target", "n_matched_runs", "gmean_error_ratio_pt_over_jax", "gmean_time_ratio_pt_over_jax"]
        if with_target
        else ["case", "n_matched_runs", "gmean_error_ratio_pt_over_jax", "gmean_time_ratio_pt_over_jax"]
    )
    result = pd.DataFrame.from_records(rows)[columns]
    sort_cols = ["case", "target"] if with_target else ["case"]
    return result.sort_values(sort_cols).reset_index(drop=True)


def build_nsb_activation_table(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[(df["pde"] == "nsb") & (df["m_train"] == M_LARGE)]

    rows = []
    for (target, activation), group in subset.groupby(["target", "activation"]):
        gmean, _ = geometric_mean_std(group["relative_test_error"].to_numpy())
        rows.append({"target": target, "activation": activation, "gmean_error": gmean})

    result = pd.DataFrame.from_records(rows)[["target", "activation", "gmean_error"]]
    return result.sort_values(["target", "activation"]).reset_index(drop=True)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_metrics()

    build_m500_table(df, "diffusion", with_target=False).to_csv(TABLES_DIR / "diffusion_table_m500.csv", index=False)
    build_dimension_table(df, "diffusion", with_target=False).to_csv(TABLES_DIR / "diffusion_table_dimension.csv", index=False)
    build_slopes_table(df, "diffusion", with_target=False).to_csv(TABLES_DIR / "diffusion_table_slopes.csv", index=False)
    build_framework_ratio_table(df, "diffusion", with_target=False).to_csv(
        TABLES_DIR / "diffusion_table_framework_ratio.csv", index=False
    )

    build_m500_table(df, "nsb", with_target=True).to_csv(TABLES_DIR / "nsb_table_m500.csv", index=False)
    build_dimension_table(df, "nsb", with_target=True).to_csv(TABLES_DIR / "nsb_table_dimension.csv", index=False)
    build_slopes_table(df, "nsb", with_target=True).to_csv(TABLES_DIR / "nsb_table_slopes.csv", index=False)
    build_framework_ratio_table(df, "nsb", with_target=True).to_csv(
        TABLES_DIR / "nsb_table_framework_ratio.csv", index=False
    )
    build_nsb_activation_table(df).to_csv(TABLES_DIR / "nsb_table_activation.csv", index=False)

    print(f"Wrote 9 tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
