"""Command-line script for PyTorch training."""

from __future__ import annotations

import argparse
import pprint

from ol_reproduction.training.pytorch_train import train_pytorch_from_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a PyTorch model on a generated dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to processed dataset directory.",
    )
    parser.add_argument(
        "--train-file",
        required=True,
        help="Training NPZ file name.",
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Testing NPZ file name.",
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

    return parser.parse_args()


def main() -> None:
    """Train model and print metrics."""
    args = parse_args()

    metrics = train_pytorch_from_files(
        dataset_dir=args.dataset,
        train_file=args.train_file,
        test_file=args.test_file,
        target=args.target,
        model_config_path=args.model,
        train_config_path=args.train,
    )

    print("\nTraining completed.")
    pprint.pp(metrics, sort_dicts=False)


if __name__ == "__main__":
    main()