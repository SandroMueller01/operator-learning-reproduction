"""Materialize flat, single-case PDE configs from the grouped configs.

configs/diffusion.yaml and configs/navier_stokes_brinkman.yaml are the
single source of truth (grouped, multi-case configs resolved via
resolve_experiment_case()). Several consumers -- run_all_available.py,
scripts/check_config.py, scripts/generate_fenics_data.py,
ol_reproduction.config.load.load_experiment_config -- expect flat,
single-case PDE config files under configs/pde/*.yaml instead. Rather than
hand-authoring (and manually keeping in sync) 8 near-duplicate files, this
script mechanically dumps resolve_experiment_case() output for every case
in both grouped configs.

The "model" and "training" top-level sections are intentionally dropped
from the materialized output: load_experiment_config() merges a PDE config
with a *separate* model config and train config file, and requires
disjoint top-level keys across the three.

Usage:
    python scripts/materialize_pde_configs.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ol_reproduction.config.load import load_yaml
from ol_reproduction.config.resolve import resolve_experiment_case

REPO_ROOT = Path(__file__).resolve().parents[1]

GROUPED_CONFIGS = {
    "diffusion": REPO_ROOT / "configs" / "diffusion.yaml",
    "nsb": REPO_ROOT / "configs" / "navier_stokes_brinkman.yaml",
}


def materialize_case(
    prefix: str,
    source_path: Path,
    grouped_config: dict[str, Any],
    case_name: str,
) -> Path:
    """Resolve one case and write it to configs/pde/<prefix>_<case_name>.yaml."""
    resolved = resolve_experiment_case(grouped_config, case_name)
    resolved.pop("model", None)
    resolved.pop("training", None)

    output_path = REPO_ROOT / "configs" / "pde" / f"{prefix}_{case_name}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# GENERATED ARTIFACT - do not hand-edit.\n"
        f"# Materialized from {source_path.relative_to(REPO_ROOT)}, "
        f"case {case_name!r}, by scripts/materialize_pde_configs.py.\n"
        "# Regenerate after changing the grouped config: "
        "python scripts/materialize_pde_configs.py\n\n"
    )

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump(resolved, handle, sort_keys=False, default_flow_style=False)

    return output_path


def main() -> None:
    written = []

    diffusion_path = GROUPED_CONFIGS["diffusion"]
    diffusion_config = load_yaml(diffusion_path)
    for case_name in diffusion_config["cases"]:
        written.append(materialize_case("diffusion", diffusion_path, diffusion_config, case_name))

    nsb_path = GROUPED_CONFIGS["nsb"]
    nsb_config = load_yaml(nsb_path)
    for case_name in nsb_config["cases"]:
        written.append(materialize_case("nsb", nsb_path, nsb_config, case_name))

    for path in written:
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    print(f"\n{len(written)} flat PDE configs materialized.")


if __name__ == "__main__":
    main()
