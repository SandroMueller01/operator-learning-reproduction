"""Check all planned config combinations."""

from __future__ import annotations

from pathlib import Path

from ol_reproduction.common.config import load_experiment_config


def main() -> None:
    """Validate all PDE, model, and training configs."""
    pde_paths = sorted(Path("configs/pde").glob("*.yaml"))
    model_paths = sorted(Path("configs/model").glob("*.yaml"))
    train_paths = sorted(Path("configs/train").glob("*.yaml"))

    num_checked = 0

    for pde_path in pde_paths:
        for model_path in model_paths:
            for train_path in train_paths:
                load_experiment_config(
                    pde_path=pde_path,
                    model_path=model_path,
                    train_path=train_path,
                )
                num_checked += 1

    print(f"All config combinations are valid. Checked: {num_checked}")


if __name__ == "__main__":
    main()