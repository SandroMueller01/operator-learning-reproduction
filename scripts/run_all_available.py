"""Run all currently available tests, data generation, training, and plots.

This script is intended as a reproducibility entry point for the practical
implementation. It only runs computations that are currently supported by the
codebase.

Available problems:
- diffusion
- navier_stokes_brinkman
- boussinesq

Available targets:
- diffusion: u
- navier_stokes_brinkman: u, p
- boussinesq: u, phi, p
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


EXPERIMENTS = (
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

MODEL_CONFIGS = (
    ModelConfig(
        name="mlp_4x40_elu",
        path="configs/model/mlp_4x40_elu.yaml",
        activation="elu",
    ),
)

PYTORCH_TRAIN_CONFIG = "configs/train/pytorch_fast_debug.yaml"
JAX_TRAIN_CONFIG = "configs/train/jax_fast_debug.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all currently available computations."
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
    """Run all available computations."""
    args = parse_args()

    _ensure_required_directories()

    if not args.skip_tests:
        _run_tests(args)

    if not args.skip_data:
        _generate_datasets(args)

    if not args.skip_pytorch:
        _run_pytorch_sweeps(args)

    if args.run_jax:
        _run_jax_sweeps(args)

    if not args.skip_plots:
        _generate_error_plots(args)

    print("\nAll requested computations finished.")


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
    )


def _generate_datasets(args: argparse.Namespace) -> None:
    """Generate datasets for all available experiments."""
    for experiment in EXPERIMENTS:
        _run_command(
            command=[
                args.python,
                "scripts/generate_data.py",
                "--pde",
                experiment.pde_config,
            ],
            dry_run=args.dry_run,
        )


def _run_pytorch_sweeps(args: argparse.Namespace) -> None:
    """Run all PyTorch sweeps."""
    for experiment in EXPERIMENTS:
        for target in experiment.targets:
            for model_config in MODEL_CONFIGS:
                output_path = _metrics_path(
                    experiment=experiment,
                    target=target,
                    framework="pytorch",
                    activation=model_config.activation,
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
                )


def _run_jax_sweeps(args: argparse.Namespace) -> None:
    """Run all JAX sweeps."""
    for experiment in EXPERIMENTS:
        for target in experiment.targets:
            for model_config in MODEL_CONFIGS:
                output_path = _metrics_path(
                    experiment=experiment,
                    target=target,
                    framework="jax",
                    activation=model_config.activation,
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
                )


def _generate_error_plots(args: argparse.Namespace) -> None:
    """Generate error-vs-m plots for all available metrics files."""
    frameworks = ["pytorch"]

    if args.run_jax:
        frameworks.append("jax")

    for experiment in EXPERIMENTS:
        for target in experiment.targets:
            for model_config in MODEL_CONFIGS:
                for framework in frameworks:
                    metrics_path = _metrics_path(
                        experiment=experiment,
                        target=target,
                        framework=framework,
                        activation=model_config.activation,
                    )
                    figure_path = _figure_path(
                        experiment=experiment,
                        target=target,
                        framework=framework,
                        activation=model_config.activation,
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
                        activation=model_config.activation,
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
                    )


def _metrics_path(
    experiment: Experiment,
    target: str,
    framework: str,
    activation: str,
) -> Path:
    """Construct metrics CSV path."""
    return Path(
        "results/metrics/"
        f"{experiment.name}_{target}_{framework}_{activation}_debug.csv"
    )


def _figure_path(
    experiment: Experiment,
    target: str,
    framework: str,
    activation: str,
) -> Path:
    """Construct figure output path."""
    return Path(
        "results/figures/"
        f"{experiment.name}_{target}_{framework}_{activation}_error_vs_m.png"
    )


def _plot_title(
    experiment: Experiment,
    target: str,
    framework: str,
    activation: str,
) -> str:
    """Construct a human-readable plot title."""
    target_names = {
        "u": "velocity/solution",
        "p": "pressure",
        "phi": "temperature",
    }

    display_target = target_names.get(target, target)
    display_framework = framework.capitalize()

    return (
        f"{experiment.name}, target={display_target}, "
        f"{display_framework} {activation.upper()}"
    )


def _ensure_required_directories() -> None:
    """Create required output directories."""
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)


def _run_command(command: list[str], dry_run: bool) -> None:
    """Run a command and fail immediately if it fails.

    Parameters
    ----------
    command:
        Command and arguments.
    dry_run:
        If true, print the command without executing it.
    """
    command_string = " ".join(command)
    print(f"\n> {command_string}")

    if dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()