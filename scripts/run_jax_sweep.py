"""Run a JAX sweep over all available training sizes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ol_reproduction.common.config import load_yaml
from ol_reproduction.evaluation.metrics_io import append_metrics_row
from ol_reproduction.training.jax_train import train_jax_from_files


TRAIN_FILE_PATTERN = re.compile(r"train_m(?P<m>\d+)_seed(?P<seed>\d+)\.npz")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run JAX training over all train_m*_seed*.npz files."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to processed dataset directory.",
    )
    parser.add_argument(
        "--target",
        default="u",
        help="Target variable to train on, e.g. u, p, or phi.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output metrics CSV.",
    )
    parser.add_argument(
        "--test-file-template",
        default="test_seed{seed}.npz",
        help="Template for test file names.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the sweep and save metrics."""
    args = parse_args()

    dataset_dir = Path(args.dataset)
    train_files = _find_train_files(dataset_dir)

    if not train_files:
        raise FileNotFoundError(
            f"No train_m*_seed*.npz files found in {dataset_dir}"
        )

    model_config = load_yaml(args.model)
    train_config = load_yaml(args.train)

    for train_file, train_size, seed in train_files:
        test_file = args.test_file_template.format(seed=seed)

        print(
            "\nRunning:",
            f"target={args.target}",
            f"m={train_size}",
            f"seed={seed}",
            f"train_file={train_file.name}",
            f"test_file={test_file}",
        )

        metrics = train_jax_from_files(
            dataset_dir=dataset_dir,
            train_file=train_file.name,
            test_file=test_file,
            target=args.target,
            model_config_path=args.model,
            train_config_path=args.train,
        )

        row = {
            "problem": _infer_problem_name(dataset_dir),
            "target": args.target,
            "framework": train_config["training"]["framework"],
            "model": model_config["model"]["name"],
            "activation": model_config["model"]["activation"],
            "depth": model_config["model"]["depth"],
            "width": model_config["model"]["width"],
            "m_train": train_size,
            "seed": seed,
            "final_train_loss": metrics["final_train_loss"],
            "relative_test_error": metrics["relative_test_error"],
            "training_time_sec": metrics["training_time_sec"],
        }

        append_metrics_row(path=args.output, row=row)
        print("Saved metrics row:", row)


def _find_train_files(dataset_dir: Path) -> list[tuple[Path, int, int]]:
    """Find and sort train files."""
    matches = []

    for path in dataset_dir.glob("train_m*_seed*.npz"):
        match = TRAIN_FILE_PATTERN.fullmatch(path.name)

        if match is None:
            continue

        train_size = int(match.group("m"))
        seed = int(match.group("seed"))
        matches.append((path, train_size, seed))

    return sorted(matches, key=lambda item: (item[1], item[2]))


def _infer_problem_name(dataset_dir: Path) -> str:
    """Infer problem name from dataset directory."""
    return dataset_dir.name


if __name__ == "__main__":
    main()