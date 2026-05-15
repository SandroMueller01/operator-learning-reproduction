"""Configuration loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


_allowed_problems = {
    "diffusion",
    "navier_stokes_brinkman",
    "boussinesq",
}

_allowed_coefficients = {
    "affine",
    "log",
}

_allowed_targets = {
    "u",
    "p",
    "phi",
}

_allowed_model_types = {
    "mlp",
}

_allowed_activations = {
    "relu",
    "elu",
    "tanh",
}

_allowed_initializations = {
    "default",
    "kaiming_uniform",
    "xavier_uniform",
}

_allowed_frameworks = {
    "pytorch",
    "jax",
}

_allowed_optimizers = {
    "adam",
}

_allowed_schedulers = {
    "none",
    "exponential_decay",
}

_allowed_losses = {
    "mse",
}

_allowed_dtypes = {
    "float32",
    "float64",
}


def load_yaml(path: str | Path) -> ConfigDict:
    """Load a YAML file as a dictionary.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the YAML file does not contain a mapping.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping: {config_path}"
        )

    return config


def merge_configs(
    pde_config: ConfigDict,
    model_config: ConfigDict,
    train_config: ConfigDict,
) -> ConfigDict:
    """Merge PDE, model, and training configs into one config dictionary.

    Parameters
    ----------
    pde_config:
        PDE configuration.
    model_config:
        Model configuration.
    train_config:
        Training configuration.

    Returns
    -------
    dict[str, Any]
        Merged experiment configuration.
    """
    return {
        "experiment": pde_config.get("experiment", {}),
        "domain": pde_config.get("domain", {}),
        "grid": pde_config.get("grid", {}),
        "pde": pde_config.get("pde", {}),
        "coefficient": pde_config.get("coefficient", {}),
        "data": pde_config.get("data", {}),
        "paths": pde_config.get("paths", {}),
        "model": model_config.get("model", {}),
        "initialization": model_config.get("initialization", {}),
        "training": train_config.get("training", {}),
        "logging": train_config.get("logging", {}),
        "reproducibility": train_config.get("reproducibility", {}),
    }


def load_experiment_config(
    pde_path: str | Path,
    model_path: str | Path,
    train_path: str | Path,
) -> ConfigDict:
    """Load, merge, and validate one experiment configuration.

    Parameters
    ----------
    pde_path:
        Path to the PDE YAML config.
    model_path:
        Path to the model YAML config.
    train_path:
        Path to the training YAML config.

    Returns
    -------
    dict[str, Any]
        Validated merged configuration.
    """
    pde_config = load_yaml(pde_path)
    model_config = load_yaml(model_path)
    train_config = load_yaml(train_path)

    config = merge_configs(
        pde_config=pde_config,
        model_config=model_config,
        train_config=train_config,
    )
    validate_config(config)

    return config


def validate_config(config: ConfigDict) -> None:
    """Validate a merged experiment configuration.

    This validator checks whether a planned experiment configuration is
    structurally valid. It intentionally accepts configs for experiments whose
    solvers may not yet be implemented, such as Navier--Stokes--Brinkman and
    Boussinesq.

    Parameters
    ----------
    config:
        Merged experiment configuration.

    Raises
    ------
    ValueError
        If a required field is missing or invalid.
    """
    _validate_required_top_level_sections(config)

    experiment = config["experiment"]
    domain = config["domain"]
    grid = config["grid"]
    pde = config["pde"]
    coefficient = config["coefficient"]
    data = config["data"]
    paths = config["paths"]
    model = config["model"]
    initialization = config["initialization"]
    training = config["training"]
    logging = config["logging"]
    reproducibility = config["reproducibility"]

    _validate_experiment_config(experiment)
    _validate_domain_config(domain)
    _validate_grid_config(grid, problem=str(experiment["problem"]))
    _validate_pde_config(pde)
    _validate_coefficient_config(coefficient)
    _validate_data_config(data)
    _validate_paths_config(paths)
    _validate_model_config(model)
    _validate_initialization_config(initialization)
    _validate_training_config(training)
    _validate_logging_config(logging)
    _validate_reproducibility_config(reproducibility)


def _validate_required_top_level_sections(config: ConfigDict) -> None:
    """Validate that all top-level sections are present."""
    required_sections = [
        "experiment",
        "domain",
        "grid",
        "pde",
        "coefficient",
        "data",
        "paths",
        "model",
        "initialization",
        "training",
        "logging",
        "reproducibility",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing top-level config section: {section}")

        if not isinstance(config[section], dict):
            raise ValueError(f"Config section '{section}' must be a mapping.")


def _validate_experiment_config(experiment: ConfigDict) -> None:
    """Validate experiment-level configuration."""
    required_keys = [
        "name",
        "problem",
        "coefficient",
        "dimension",
    ]

    for key in required_keys:
        if key not in experiment:
            raise ValueError(f"Missing experiment.{key}")

    problem = str(experiment["problem"])
    coefficient = str(experiment["coefficient"])

    if problem not in _allowed_problems:
        raise ValueError(
            "experiment.problem must be one of "
            f"{sorted(_allowed_problems)}, got {problem!r}."
        )

    if coefficient not in _allowed_coefficients:
        raise ValueError(
            "experiment.coefficient must be one of "
            f"{sorted(_allowed_coefficients)}, got {coefficient!r}."
        )

    if int(experiment["dimension"]) <= 0:
        raise ValueError("experiment.dimension must be positive.")

    _validate_targets(experiment)


def _validate_targets(experiment: ConfigDict) -> None:
    """Validate single-output or multi-output target configuration."""
    has_target = "target" in experiment
    has_targets = "targets" in experiment

    if has_target and has_targets:
        raise ValueError(
            "Use either experiment.target or experiment.targets, not both."
        )

    if not has_target and not has_targets:
        raise ValueError("Missing experiment.target or experiment.targets.")

    if has_target:
        target = str(experiment["target"])

        if target not in _allowed_targets:
            raise ValueError(
                "experiment.target must be one of "
                f"{sorted(_allowed_targets)}, got {target!r}."
            )

    if has_targets:
        targets = experiment["targets"]

        if not isinstance(targets, list):
            raise ValueError("experiment.targets must be a list.")

        if not targets:
            raise ValueError("experiment.targets must not be empty.")

        for target in targets:
            target_name = str(target)

            if target_name not in _allowed_targets:
                raise ValueError(
                    "All experiment.targets must be one of "
                    f"{sorted(_allowed_targets)}, got {target_name!r}."
                )


def _validate_domain_config(domain: ConfigDict) -> None:
    """Validate domain configuration."""
    if "type" not in domain:
        raise ValueError("Missing domain.type.")

    domain_type = str(domain["type"])

    if domain_type == "unit_square":
        _require_numeric_domain_keys(
            domain=domain,
            keys=["x_min", "x_max", "y_min", "y_max"],
        )
        _validate_interval(domain["x_min"], domain["x_max"], "x")
        _validate_interval(domain["y_min"], domain["y_max"], "y")
        return

    if domain_type == "unit_cube":
        _require_numeric_domain_keys(
            domain=domain,
            keys=[
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "z_min",
                "z_max",
            ],
        )
        _validate_interval(domain["x_min"], domain["x_max"], "x")
        _validate_interval(domain["y_min"], domain["y_max"], "y")
        _validate_interval(domain["z_min"], domain["z_max"], "z")
        return

    raise ValueError(
        "domain.type must be either 'unit_square' or 'unit_cube', "
        f"got {domain_type!r}."
    )


def _require_numeric_domain_keys(
    domain: ConfigDict,
    keys: list[str],
) -> None:
    """Validate required numeric domain keys."""
    for key in keys:
        if key not in domain:
            raise ValueError(f"Missing domain.{key}")

        _require_numeric_value(domain[key], f"domain.{key}")


def _validate_interval(
    minimum: Any,
    maximum: Any,
    axis_name: str,
) -> None:
    """Validate that an interval is ordered correctly."""
    if float(minimum) >= float(maximum):
        raise ValueError(
            f"domain.{axis_name}_min must be smaller than "
            f"domain.{axis_name}_max."
        )


def _validate_grid_config(
    grid: ConfigDict,
    problem: str,
) -> None:
    """Validate grid configuration."""
    if problem in {"diffusion", "navier_stokes_brinkman"}:
        required_keys = ["nx", "ny"]
    elif problem == "boussinesq":
        required_keys = ["nx", "ny", "nz"]
    else:
        raise ValueError(f"Unsupported problem: {problem}")

    for key in required_keys:
        if key not in grid:
            raise ValueError(f"Missing grid.{key}")

        if int(grid[key]) < 4:
            raise ValueError(f"grid.{key} must be at least 4.")

    if "include_boundary" in grid and not isinstance(
        grid["include_boundary"],
        bool,
    ):
        raise ValueError("grid.include_boundary must be boolean.")


def _validate_pde_config(pde: ConfigDict) -> None:
    """Validate PDE configuration."""
    if "equation" not in pde:
        raise ValueError("Missing pde.equation.")

    if "boundary_conditions" not in pde:
        raise ValueError("Missing pde.boundary_conditions.")

    boundary_conditions = pde["boundary_conditions"]

    if not isinstance(boundary_conditions, dict):
        raise ValueError("pde.boundary_conditions must be a mapping.")

    if "type" not in boundary_conditions:
        raise ValueError("Missing pde.boundary_conditions.type.")

    if pde["equation"] == "elliptic_diffusion":
        if "forcing" not in pde:
            raise ValueError("Missing pde.forcing for elliptic_diffusion.")

        _require_numeric_value(pde["forcing"], "pde.forcing")

        required_boundary_keys = ["bottom", "top", "left", "right"]

        for key in required_boundary_keys:
            if key not in boundary_conditions:
                raise ValueError(f"Missing pde.boundary_conditions.{key}")

            _require_numeric_value(
                boundary_conditions[key],
                f"pde.boundary_conditions.{key}",
            )


def _validate_coefficient_config(coefficient: ConfigDict) -> None:
    """Validate coefficient configuration."""
    required_keys = [
        "name",
        "base_value",
    ]

    for key in required_keys:
        if key not in coefficient:
            raise ValueError(f"Missing coefficient.{key}")

    coefficient_name = str(coefficient["name"])

    if coefficient_name not in _allowed_coefficients:
        raise ValueError(
            "coefficient.name must be one of "
            f"{sorted(_allowed_coefficients)}, got {coefficient_name!r}."
        )

    _require_numeric_value(coefficient["base_value"], "coefficient.base_value")

    if float(coefficient["base_value"]) <= 0.0:
        raise ValueError("coefficient.base_value must be positive.")


def _validate_data_config(data: ConfigDict) -> None:
    """Validate data-generation configuration."""
    required_keys = [
        "train_sizes",
        "test_size",
        "seeds",
        "dtype",
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing data.{key}")

    train_sizes = data["train_sizes"]
    seeds = data["seeds"]
    dtype = str(data["dtype"])

    if not isinstance(train_sizes, list):
        raise ValueError("data.train_sizes must be a list.")

    if not train_sizes:
        raise ValueError("data.train_sizes must not be empty.")

    for train_size in train_sizes:
        if int(train_size) <= 0:
            raise ValueError("All data.train_sizes must be positive.")

    if int(data["test_size"]) <= 0:
        raise ValueError("data.test_size must be positive.")

    if not isinstance(seeds, list):
        raise ValueError("data.seeds must be a list.")

    if not seeds:
        raise ValueError("data.seeds must not be empty.")

    for seed in seeds:
        if int(seed) < 0:
            raise ValueError("All data.seeds must be non-negative.")

    if dtype not in _allowed_dtypes:
        raise ValueError(
            "data.dtype must be one of "
            f"{sorted(_allowed_dtypes)}, got {dtype!r}."
        )


def _validate_paths_config(paths: ConfigDict) -> None:
    """Validate paths configuration."""
    if "output_dir" not in paths:
        raise ValueError("Missing paths.output_dir.")

    output_dir = paths["output_dir"]

    if not isinstance(output_dir, str):
        raise ValueError("paths.output_dir must be a string.")

    if not output_dir:
        raise ValueError("paths.output_dir must not be empty.")


def _validate_model_config(model: ConfigDict) -> None:
    """Validate model configuration."""
    required_keys = [
        "name",
        "type",
        "depth",
        "width",
        "activation",
    ]

    for key in required_keys:
        if key not in model:
            raise ValueError(f"Missing model.{key}")

    model_type = str(model["type"])
    activation = str(model["activation"])

    if model_type not in _allowed_model_types:
        raise ValueError(
            "model.type must be one of "
            f"{sorted(_allowed_model_types)}, got {model_type!r}."
        )

    if int(model["depth"]) <= 0:
        raise ValueError("model.depth must be positive.")

    if int(model["width"]) <= 0:
        raise ValueError("model.width must be positive.")

    if activation not in _allowed_activations:
        raise ValueError(
            "model.activation must be one of "
            f"{sorted(_allowed_activations)}, got {activation!r}."
        )


def _validate_initialization_config(initialization: ConfigDict) -> None:
    """Validate initialization configuration."""
    if "name" not in initialization:
        raise ValueError("Missing initialization.name.")

    initialization_name = str(initialization["name"])

    if initialization_name not in _allowed_initializations:
        raise ValueError(
            "initialization.name must be one of "
            f"{sorted(_allowed_initializations)}, "
            f"got {initialization_name!r}."
        )


def _validate_training_config(training: ConfigDict) -> None:
    """Validate training configuration."""
    required_keys = [
        "framework",
        "optimizer",
        "scheduler",
        "batch_size",
        "epochs",
        "early_stopping",
        "loss",
        "device",
        "dtype",
    ]

    for key in required_keys:
        if key not in training:
            raise ValueError(f"Missing training.{key}")

    framework = str(training["framework"])
    dtype = str(training["dtype"])

    if framework not in _allowed_frameworks:
        raise ValueError(
            "training.framework must be one of "
            f"{sorted(_allowed_frameworks)}, got {framework!r}."
        )

    if int(training["epochs"]) <= 0:
        raise ValueError("training.epochs must be positive.")

    if dtype not in _allowed_dtypes:
        raise ValueError(
            "training.dtype must be one of "
            f"{sorted(_allowed_dtypes)}, got {dtype!r}."
        )

    _validate_optimizer_config(training["optimizer"])
    _validate_scheduler_config(training["scheduler"])
    _validate_early_stopping_config(training["early_stopping"])
    _validate_loss_config(training["loss"])


def _validate_optimizer_config(optimizer: Any) -> None:
    """Validate optimizer configuration."""
    if not isinstance(optimizer, dict):
        raise ValueError("training.optimizer must be a mapping.")

    required_keys = [
        "name",
        "learning_rate",
    ]

    for key in required_keys:
        if key not in optimizer:
            raise ValueError(f"Missing training.optimizer.{key}")

    optimizer_name = str(optimizer["name"])

    if optimizer_name not in _allowed_optimizers:
        raise ValueError(
            "training.optimizer.name must be one of "
            f"{sorted(_allowed_optimizers)}, got {optimizer_name!r}."
        )

    if float(optimizer["learning_rate"]) <= 0.0:
        raise ValueError("training.optimizer.learning_rate must be positive.")

    if "weight_decay" in optimizer and float(optimizer["weight_decay"]) < 0.0:
        raise ValueError("training.optimizer.weight_decay must be non-negative.")


def _validate_scheduler_config(scheduler: Any) -> None:
    """Validate scheduler configuration."""
    if not isinstance(scheduler, dict):
        raise ValueError("training.scheduler must be a mapping.")

    if "name" not in scheduler:
        raise ValueError("Missing training.scheduler.name.")

    scheduler_name = str(scheduler["name"])

    if scheduler_name not in _allowed_schedulers:
        raise ValueError(
            "training.scheduler.name must be one of "
            f"{sorted(_allowed_schedulers)}, got {scheduler_name!r}."
        )

    if scheduler_name == "exponential_decay":
        required_keys = [
            "decay_rate",
            "decay_steps",
        ]

        for key in required_keys:
            if key not in scheduler:
                raise ValueError(f"Missing training.scheduler.{key}")

        if float(scheduler["decay_rate"]) <= 0.0:
            raise ValueError("training.scheduler.decay_rate must be positive.")

        if int(scheduler["decay_steps"]) <= 0:
            raise ValueError("training.scheduler.decay_steps must be positive.")


def _validate_early_stopping_config(early_stopping: Any) -> None:
    """Validate early stopping configuration."""
    if not isinstance(early_stopping, dict):
        raise ValueError("training.early_stopping must be a mapping.")

    required_keys = [
        "enabled",
        "patience",
        "min_delta",
    ]

    for key in required_keys:
        if key not in early_stopping:
            raise ValueError(f"Missing training.early_stopping.{key}")

    if not isinstance(early_stopping["enabled"], bool):
        raise ValueError("training.early_stopping.enabled must be boolean.")

    if int(early_stopping["patience"]) < 0:
        raise ValueError("training.early_stopping.patience must be non-negative.")

    if float(early_stopping["min_delta"]) < 0.0:
        raise ValueError("training.early_stopping.min_delta must be non-negative.")


def _validate_loss_config(loss: Any) -> None:
    """Validate loss configuration."""
    if not isinstance(loss, dict):
        raise ValueError("training.loss must be a mapping.")

    if "name" not in loss:
        raise ValueError("Missing training.loss.name.")

    loss_name = str(loss["name"])

    if loss_name not in _allowed_losses:
        raise ValueError(
            "training.loss.name must be one of "
            f"{sorted(_allowed_losses)}, got {loss_name!r}."
        )


def _validate_logging_config(logging: ConfigDict) -> None:
    """Validate logging configuration."""
    required_keys = [
        "log_every",
        "save_checkpoint",
        "save_metrics",
    ]

    for key in required_keys:
        if key not in logging:
            raise ValueError(f"Missing logging.{key}")

    if int(logging["log_every"]) <= 0:
        raise ValueError("logging.log_every must be positive.")

    if not isinstance(logging["save_checkpoint"], bool):
        raise ValueError("logging.save_checkpoint must be boolean.")

    if not isinstance(logging["save_metrics"], bool):
        raise ValueError("logging.save_metrics must be boolean.")


def _validate_reproducibility_config(reproducibility: ConfigDict) -> None:
    """Validate reproducibility configuration."""
    if "deterministic" not in reproducibility:
        raise ValueError("Missing reproducibility.deterministic.")

    if not isinstance(reproducibility["deterministic"], bool):
        raise ValueError("reproducibility.deterministic must be boolean.")

    if "seed" in reproducibility and int(reproducibility["seed"]) < 0:
        raise ValueError("reproducibility.seed must be non-negative.")


def _require_numeric_value(value: Any, name: str) -> None:
    """Validate that a value can be converted to float.

    Parameters
    ----------
    value:
        Value to validate.
    name:
        Config field name for error messages.

    Raises
    ------
    ValueError
        If the value is not numeric.
    """
    try:
        float(value)
    except TypeError as error:
        raise ValueError(f"{name} must be numeric.") from error
    except ValueError as error:
        raise ValueError(f"{name} must be numeric.") from error