"""Run available tests, data generation, training sweeps, and plots.

This script is intended as a reproducibility entry point for the practical
implementation.

Default mode runs a small debug matrix:
- diffusion_affine_d4, target u
- nsb_affine_d4, targets u and p
- boussinesq_affine_d4, targets u, phi and p
- model mlp_4x40_elu

Paper-matrix mode expands to:
- affine/log coefficients
- d=4/d=8
- all three PDE families
- all available targets
- 4x40 and 10x100 MLPs
- ReLU, ELU and tanh
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    """Experiment definition for one dataset and its trainable targets."""

    name: str
    pde_config: str
    dataset_dir: str
    targets: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration definition."""

    name: str
    path: str
    activation: str


DEBUG_EXPERIMENTS = (
    Experiment(
        name="diffusion_affine_d4",
        pde_config="configs/pde/diffusion_affine_d4.yaml",
        dataset_dir="data/processed/diffusion_affine_d4",
        targets=("u",),
    ),
    Experiment(
        name="nsb_affine_d4",
        pde_config="configs/pde/nsb_affine_d4.yaml",
        dataset_dir="data/processed/nsb_affine_d4",
        targets=("u", "p"),
    ),
    Experiment(
        name="boussinesq_affine_d4",
        pde_config="configs/pde/boussinesq_affine_d4.yaml",
        dataset_dir="data/processed/boussinesq_affine_d4",
        targets=("u", "phi", "p"),
    ),
)

PAPER_EXPERIMENTS = (
    Experiment(
        name="diffusion_affine_d4",
        pde_config="configs/pde/diffusion_affine_d4.yaml",
        dataset_dir="data/processed/diffusion_affine_d4",
        targets=("u",),
    ),
    Experiment(
        name="diffusion_affine_d8",
        pde_config="configs/pde/diffusion_affine_d8.yaml",
        dataset_dir="data/processed/diffusion_affine_d8",
        targets=("u",),
    ),
    Experiment(
        name="diffusion_log_d4",
        pde_config="configs/pde/diffusion_log_d4.yaml",
        dataset_dir="data/processed/diffusion_log_d4",
        targets=("u",),
    ),
    Experiment(
        name="diffusion_log_d8",
        pde_config="configs/pde/diffusion_log_d8.yaml",
        dataset_dir="data/processed/diffusion_log_d8",
        targets=("u",),
    ),
    Experiment(
        name="nsb_affine_d4",
        pde_config="configs/pde/nsb_affine_d4.yaml",
        dataset_dir="data/processed/nsb_affine_d4",
        targets=("u", "p"),
    ),
    Experiment(
        name="nsb_affine_d8",
        pde_config="configs/pde/nsb_affine_d8.yaml",
        dataset_dir="data/processed/nsb_affine_d8",
        targets=("u", "p"),
    ),
    Experiment(
        name="nsb_log_d4",
        pde_config="configs/pde/nsb_log_d4.yaml",
        dataset_dir="data/processed/nsb_log_d4",
        targets=("u", "p"),
    ),
    Experiment(
        name="nsb_log_d8",
        pde_config="configs/pde/nsb_log_d8.yaml",
        dataset_dir="data/processed/nsb_log_d8",
        targets=("u", "p"),
    ),
    Experiment(
        name="boussinesq_affine_d4",
        pde_config="configs/pde/boussinesq_affine_d4.yaml",
        dataset_dir="data/processed/boussinesq_affine_d4",
        targets=("u", "phi", "p"),
    ),
    Experiment(
        name="boussinesq_affine_d8",
        pde_config="configs/pde/boussinesq_affine_d8.yaml",
        dataset_dir="data/processed/boussinesq_affine_d8",
        targets=("u", "phi", "p"),
    ),
    Experiment(
        name="boussinesq_log_d4",
        pde_config="configs/pde/boussinesq_log_d4.yaml",
        dataset_dir="data/processed/boussinesq_log_d4",
        targets=("u", "phi", "p"),
    ),
    Experiment(
        name="boussinesq_log_d8",
        pde_config="configs/pde/boussinesq_log_d8.yaml",
        dataset_dir="data/processed/boussinesq_log_d8",
        targets=("u", "phi", "p"),
    ),
)

DEBUG_MODEL_CONFIGS = (
    ModelConfig(
        name="mlp_4x40_elu",
        path="configs/model/mlp_4x40_elu.yaml",
        activation="elu",
    ),
)

PAPER_MODEL_CONFIGS = (
    ModelConfig(
        name="mlp_4x40_relu",
        path="configs/model/mlp_4x40_relu.yaml",
        activation="relu",
    ),
    ModelConfig(
        name="mlp_4x40_elu",
        path="configs/model/mlp_4x40_elu.yaml",
        activation="elu",
    ),
    ModelConfig(
        name="mlp_4x40_tanh",
        path="configs/model/mlp_4x40_tanh.yaml",
        activation="tanh",
    ),
    ModelConfig(
        name="mlp_10x100_relu",
        path="configs/model/mlp_10x100_relu.yaml",
        activation="relu",
    ),
    ModelConfig(
        name="mlp_10x100_elu",
        path="configs/model/mlp_10x100_elu.yaml",
        activation="elu",
    ),
    ModelConfig(
        name="mlp_10x100_tanh",
        path="configs/model/mlp_10x100_tanh.yaml",
        activation="tanh",
    ),
)

PYTORCH_TRAIN_CONFIG = "configs/train/pytorch_fast_debug.yaml"
JAX_TRAIN_CONFIG = "configs/train/jax_fast_debug.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run available computations for the reproduction project."
    )
    parser.add_argument(
        "--paper-matrix",
        action="store_true",
        help=(
            "Run the larger paper-like matrix instead of the small debug "
            "matrix. This can be expensive."
        ),
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest execution.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip dataset generation.",
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch sweeps.",
    )
    parser.add_argument(
        "--run-jax",
        action="store_true",
        help="Also run JAX sweeps. Disabled by default to save time.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip error-vs-m plot generation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining commands if one command fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Defaults to current interpreter.",
    )

    return parser.parse_args()


def main() -> None:
    """Run requested computations."""
    args = parse_args()

    experiments = _select_experiments(args)
    model_configs = _select_model_configs(args)

    _ensure_required_directories()

    _print_run_summary(
        args=args,
        experiments=experiments,
        model_configs=model_configs,
    )

    if not args.skip_tests:
        _run_tests(args)

    if not args.skip_data:
        _generate_datasets(
            args=args,
            experiments=experiments,
        )

    if not args.skip_pytorch:
        _run_pytorch_sweeps(
            args=args,
            experiments=experiments,
            model_configs=model_configs,
        )

    if args.run_jax:
        _run_jax_sweeps(
            args=args,
            experiments=experiments,
            model_configs=model_configs,
        )

    if not args.skip_plots:
        _generate_error_plots(
            args=args,
            experiments=experiments,
            model_configs=model_configs,
        )

    print("\nAll requested computations finished.")


def _select_experiments(args: argparse.Namespace) -> tuple[Experiment, ...]:
    """Select experiment matrix."""
    if args.paper_matrix:
        return PAPER_EXPERIMENTS

    return DEBUG_EXPERIMENTS


def _select_model_configs(args: argparse.Namespace) -> tuple[ModelConfig, ...]:
    """Select model matrix."""
    if args.paper_matrix:
        return PAPER_MODEL_CONFIGS

    return DEBUG_MODEL_CONFIGS


def _print_run_summary(
    args: argparse.Namespace,
    experiments: tuple[Experiment, ...],
    model_configs: tuple[ModelConfig, ...],
) -> None:
    """Print run summary before executing commands."""
    num_targets = sum(len(experiment.targets) for experiment in experiments)
    num_frameworks = 1 + int(args.run_jax)

    print("\nRun configuration:")
    print(f"  paper_matrix: {args.paper_matrix}")
    print(f"  experiments: {len(experiments)}")
    print(f"  target-experiments: {num_targets}")
    print(f"  models: {len(model_configs)}")
    print(f"  frameworks: {num_frameworks}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  continue_on_error: {args.continue_on_error}")

    if args.paper_matrix:
        print(
            "\nWarning: --paper-matrix can be expensive. "
            "Use --dry-run first to inspect the command list."
        )


def _run_tests(args: argparse.Namespace) -> None:
    """Run pytest."""
    _run_command(
        command=[
            args.python,
            "-m",
            "pytest",
            "-q",
            "tests",
        ],
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


def _generate_datasets(
    args: argparse.Namespace,
    experiments: tuple[Experiment, ...],
) -> None:
    """Generate datasets for selected experiments."""
    for experiment in experiments:
        _run_command(
            command=[
                args.python,
                "scripts/generate_data.py",
                "--pde",
                experiment.pde_config,
            ],
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )


def _run_pytorch_sweeps(
    args: argparse.Namespace,
    experiments: tuple[Experiment, ...],
    model_configs: tuple[ModelConfig, ...],
) -> None:
    """Run selected PyTorch sweeps."""
    for experiment in experiments:
        for target in experiment.targets:
            for model_config in model_configs:
                output_path = _metrics_path(
                    experiment=experiment,
                    target=target,
                    framework="pytorch",
                    model_name=model_config.name,
                )

                _run_command(
                    command=[
                        args.python,
                        "scripts/run_pytorch_sweep.py",
                        "--dataset",
                        experiment.dataset_dir,
                        "--target",
                        target,
                        "--model",
                        model_config.path,
                        "--train",
                        PYTORCH_TRAIN_CONFIG,
                        "--output",
                        str(output_path),
                    ],
                    dry_run=args.dry_run,
                    continue_on_error=args.continue_on_error,
                )


def _run_jax_sweeps(
    args: argparse.Namespace,
    experiments: tuple[Experiment, ...],
    model_configs: tuple[ModelConfig, ...],
) -> None:
    """Run selected JAX sweeps."""
    for experiment in experiments:
        for target in experiment.targets:
            for model_config in model_configs:
                output_path = _metrics_path(
                    experiment=experiment,
                    target=target,
                    framework="jax",
                    model_name=model_config.name,
                )

                _run_command(
                    command=[
                        args.python,
                        "scripts/run_jax_sweep.py",
                        "--dataset",
                        experiment.dataset_dir,
                        "--target",
                        target,
                        "--model",
                        model_config.path,
                        "--train",
                        JAX_TRAIN_CONFIG,
                        "--output",
                        str(output_path),
                    ],
                    dry_run=args.dry_run,
                    continue_on_error=args.continue_on_error,
                )


def _generate_error_plots(
    args: argparse.Namespace,
    experiments: tuple[Experiment, ...],
    model_configs: tuple[ModelConfig, ...],
) -> None:
    """Generate error-vs-m plots for selected metrics files."""
    frameworks = ["pytorch"]

    if args.run_jax:
        frameworks.append("jax")

    for experiment in experiments:
        for target in experiment.targets:
            for model_config in model_configs:
                for framework in frameworks:
                    metrics_path = _metrics_path(
                        experiment=experiment,
                        target=target,
                        framework=framework,
                        model_name=model_config.name,
                    )
                    figure_path = _figure_path(
                        experiment=experiment,
                        target=target,
                        framework=framework,
                        model_name=model_config.name,
                    )

                    if not metrics_path.exists() and not args.dry_run:
                        print(
                            "Skipping plot because metrics file does not "
                            f"exist: {metrics_path}"
                        )
                        continue

                    title = _plot_title(
                        experiment=experiment,
                        target=target,
                        framework=framework,
                        model_name=model_config.name,
                    )

                    _run_command(
                        command=[
                            args.python,
                            "scripts/plot_error_vs_m.py",
                            "--metrics",
                            str(metrics_path),
                            "--output",
                            str(figure_path),
                            "--title",
                            title,
                        ],
                        dry_run=args.dry_run,
                        continue_on_error=args.continue_on_error,
                    )


def _metrics_path(
    experiment: Experiment,
    target: str,
    framework: str,
    model_name: str,
) -> Path:
    """Construct metrics CSV path.

    Parameters
    ----------
    experiment:
        Experiment definition.
    target:
        Target name.
    framework:
        Training framework name.
    model_name:
        Full model config name.

    Returns
    -------
    Path
        Metrics CSV path.
    """
    return Path(
        "results/metrics/"
        f"{experiment.name}_{target}_{framework}_{model_name}_debug.csv"
    )


def _figure_path(
    experiment: Experiment,
    target: str,
    framework: str,
    model_name: str,
) -> Path:
    """Construct figure output path.

    Parameters
    ----------
    experiment:
        Experiment definition.
    target:
        Target name.
    framework:
        Training framework name.
    model_name:
        Full model config name.

    Returns
    -------
    Path
        Figure path.
    """
    return Path(
        "results/figures/"
        f"{experiment.name}_{target}_{framework}_{model_name}_error_vs_m.png"
    )


def _plot_title(
    experiment: Experiment,
    target: str,
    framework: str,
    model_name: str,
) -> str:
    """Construct a human-readable plot title.

    Parameters
    ----------
    experiment:
        Experiment definition.
    target:
        Target name.
    framework:
        Training framework name.
    model_name:
        Full model config name.

    Returns
    -------
    str
        Plot title.
    """
    target_names = {
        "u": "velocity/solution",
        "p": "pressure",
        "phi": "temperature",
    }

    display_target = target_names.get(target, target)
    display_framework = framework.capitalize()

    return (
        f"{experiment.name}, target={display_target}, "
        f"{display_framework}, {model_name}"
    )


def _ensure_required_directories() -> None:
    """Create required output directories."""
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)


def _run_command(
    command: list[str],
    dry_run: bool,
    continue_on_error: bool,
) -> None:
    """Run a command.

    Parameters
    ----------
    command:
        Command and arguments.
    dry_run:
        If true, print the command without executing it.
    continue_on_error:
        If true, continue after failed commands.
    """
    command_string = " ".join(command)
    print(f"\n> {command_string}")

    if dry_run:
        return

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        if continue_on_error:
            print(
                "Command failed but continuing because "
                "--continue-on-error was set."
            )
            print(f"Exit code: {error.returncode}")
            return

        raise


if __name__ == "__main__":
    main()