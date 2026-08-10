"""Resolve grouped experiment configs into concrete experiment configs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

ConfigDict = dict[str, Any]


def resolve_experiment_case(config: ConfigDict, case_name: str) -> ConfigDict:
    """Resolve one case from a grouped experiment config.

    Grouped configs contain shared PDE settings and several cases, such as
    ``affine_d4`` or ``log_d8``. This function selects one case and merges it
    with the shared settings so the rest of the pipeline receives one concrete
    experiment config.
    """
    if "experiment" not in config:
        raise ValueError("Config is missing required 'experiment' section.")

    if "cases" not in config:
        raise ValueError("Config is missing required 'cases' section.")

    if "coefficients" not in config:
        raise ValueError("Config is missing required 'coefficients' section.")

    cases = config["cases"]

    if not isinstance(cases, dict):
        raise ValueError("Config section 'cases' must be a mapping.")

    if case_name not in cases:
        available_cases = sorted(cases)
        raise ValueError(
            f"Unknown case {case_name!r}. Available cases are: "
            f"{available_cases}."
        )

    case_config = cases[case_name]

    required_case_keys = [
        "name",
        "coefficient",
        "dimension",
        "targets",
        "paths",
    ]
    for key in required_case_keys:
        if key not in case_config:
            raise ValueError(f"Missing cases.{case_name}.{key}.")

    coefficient_name = str(case_config["coefficient"])
    coefficients = config["coefficients"]

    if coefficient_name not in coefficients:
        available_coefficients = sorted(coefficients)
        raise ValueError(
            f"Unknown coefficient {coefficient_name!r}. Available "
            f"coefficients are: {available_coefficients}."
        )

    coefficient_config = deepcopy(coefficients[coefficient_name])

    resolved = {
        "experiment": {
            "name": str(case_config["name"]),
            "group_name": str(config["experiment"]["group_name"]),
            "problem": str(config["experiment"]["problem"]),
            "equation": str(config["experiment"]["equation"]),
            "coefficient": coefficient_name,
            "dimension": int(case_config["dimension"]),
            "targets": list(case_config["targets"]),
            "case": case_name,
        },
        "pde": deepcopy(config["pde"]),
        "coefficient": coefficient_config,
        "discretization": deepcopy(config["discretization"]),
        "data": deepcopy(config["data"]),
        "testing": deepcopy(config.get("testing", {})),
        "model": deepcopy(config.get("model", {})),
        "training": deepcopy(config.get("training", {})),
        "evaluation": deepcopy(config.get("evaluation", {})),
        "paths": deepcopy(case_config["paths"]),
        "reproduction": deepcopy(config.get("reproduction", {})),
    }

    return resolved