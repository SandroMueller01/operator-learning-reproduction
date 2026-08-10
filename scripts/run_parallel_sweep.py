"""Run the training sweep with many concurrent worker processes.

scripts/run_pytorch_sweep.py and scripts/run_jax_sweep.py each run one
(experiment, target, model) sweep serially, one (m, seed) combination at a
time. At the paper's full matrix scale this is thousands of independent,
short (tens of seconds to a few minutes) training runs -- an embarrassingly
parallel workload. Measured single-run timings (Phase 11 timing pilot) put
a fully serial run of the whole matrix at roughly two weeks on this
machine; this script instead fans the same task list out across a process
pool so the wall-clock time scales down with the number of CPU cores
actually used, without changing anything about what is trained (same
configs, same trial seeds, same checkpoint/restore rule).

GPU offered no benefit for these small full-batch MLPs (see
configs/train/pytorch_paper_parallel.yaml), so PyTorch tasks are pinned to
CPU explicitly and each worker is limited to a small BLAS thread count to
avoid oversubscribing the machine's cores across many concurrent processes.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_all_available import (  # noqa: E402
    DEBUG_EXPERIMENTS,
    DEBUG_JAX_TRAIN_CONFIG,
    DEBUG_MODEL_CONFIGS,
    DEBUG_PYTORCH_TRAIN_CONFIG,
    PAPER_EXPERIMENTS,
    PAPER_JAX_TRAIN_CONFIG,
    PAPER_MODEL_CONFIGS,
    Experiment,
    ModelConfig,
    _metrics_path,
)

TRAIN_FILE_PATTERN = re.compile(r"train_m(?P<m>\d+)_seed(?P<seed>\d+)\.npz")
PYTORCH_PAPER_PARALLEL_TRAIN_CONFIG = "configs/train/pytorch_paper_parallel.yaml"

# Documented deviation (see configs/train/pytorch_nsb.yaml): NSB tasks under
# the paper's full 60,000-epoch / 5e-7-tolerance protocol were measured at
# ~9 days of wall-clock time for the full matrix even at 16-way parallelism,
# so NSB specifically uses a reduced-budget protocol. Diffusion is
# unaffected and keeps the paper's exact protocol.
NSB_PYTORCH_TRAIN_CONFIG = "configs/train/pytorch_nsb.yaml"
NSB_JAX_TRAIN_CONFIG = "configs/train/jax_nsb.yaml"

DEFAULT_BLAS_THREADS = "1"


@dataclass(frozen=True)
class Task:
    """One (experiment, target, model, framework, m, seed) training run."""

    framework: str
    dataset_dir: str
    target: str
    model_path: str
    model_name: str
    train_config_path: str
    train_file: str
    m_train: int
    seed: int
    output_csv: str
    problem: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the PyTorch/JAX training sweep across a process pool."
    )
    parser.add_argument(
        "--paper-matrix",
        action="store_true",
        help="Use the full paper experiment/model matrix instead of the debug one.",
    )
    parser.add_argument(
        "--framework",
        choices=["pytorch", "jax", "both"],
        default="both",
        help="Which framework(s) to run.",
    )
    parser.add_argument(
        "--case-subset",
        nargs="+",
        default=None,
        metavar="EXPERIMENT_NAME",
        help="Restrict to specific experiment names, e.g. diffusion_affine_d4.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 12),
        help=(
            "Number of concurrent worker processes. Defaults to "
            "cpu_count - 12, leaving headroom for the background WSL2 "
            "data-generation job and the OS."
        ),
    )
    parser.add_argument(
        "--blas-threads",
        default=DEFAULT_BLAS_THREADS,
        help="BLAS/torch intra-op thread count per worker (kept low to avoid oversubscription).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (m, seed) rows already present in each task's output CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the task count/summary and exit without training anything.",
    )

    return parser.parse_args()


def main() -> None:
    """Build the task list and run it across a process pool."""
    args = parse_args()

    experiments = PAPER_EXPERIMENTS if args.paper_matrix else DEBUG_EXPERIMENTS
    model_configs = PAPER_MODEL_CONFIGS if args.paper_matrix else DEBUG_MODEL_CONFIGS

    if args.case_subset is not None:
        selected = set(args.case_subset)
        experiments = tuple(e for e in experiments if e.name in selected)

    frameworks = ["pytorch", "jax"] if args.framework == "both" else [args.framework]

    tasks = _build_tasks(
        experiments=experiments,
        model_configs=model_configs,
        frameworks=frameworks,
        paper_matrix=args.paper_matrix,
        resume=args.resume,
        pytorch_train_config=(
            PYTORCH_PAPER_PARALLEL_TRAIN_CONFIG if args.paper_matrix else DEBUG_PYTORCH_TRAIN_CONFIG
        ),
        jax_train_config=PAPER_JAX_TRAIN_CONFIG if args.paper_matrix else DEBUG_JAX_TRAIN_CONFIG,
    )

    print(
        f"\nTotal tasks: {len(tasks)} "
        f"(experiments={len(experiments)}, models={len(model_configs)}, "
        f"frameworks={frameworks}, workers={args.workers})"
    )
    by_framework: dict[str, int] = {}
    for task in tasks:
        by_framework[task.framework] = by_framework.get(task.framework, 0) + 1
    print("Tasks per framework:", by_framework)

    if args.dry_run or not tasks:
        return

    _run_pool(tasks=tasks, workers=args.workers, blas_threads=args.blas_threads)


def _build_tasks(
    experiments: tuple[Experiment, ...],
    model_configs: tuple[ModelConfig, ...],
    frameworks: list[str],
    paper_matrix: bool,
    resume: bool,
    pytorch_train_config: str,
    jax_train_config: str,
) -> list[Task]:
    """Enumerate every (experiment, target, model, framework, m, seed) task."""
    tasks: list[Task] = []

    for experiment in experiments:
        dataset_dir = Path(experiment.dataset_dir)

        if not dataset_dir.exists():
            print(f"Skipping {experiment.name}: dataset not generated yet ({dataset_dir}).")
            continue

        train_files = _find_train_files(dataset_dir)
        if not train_files:
            print(f"Skipping {experiment.name}: no train_m*_seed*.npz files found.")
            continue

        for target in experiment.targets:
            for model_config in model_configs:
                for framework in frameworks:
                    if experiment.name.startswith("nsb") and paper_matrix:
                        train_config_path = (
                            NSB_PYTORCH_TRAIN_CONFIG if framework == "pytorch" else NSB_JAX_TRAIN_CONFIG
                        )
                    else:
                        train_config_path = (
                            pytorch_train_config if framework == "pytorch" else jax_train_config
                        )
                    output_csv = str(
                        _metrics_path(
                            experiment=experiment,
                            target=target,
                            framework=framework,
                            model_name=model_config.name,
                            paper_matrix=paper_matrix,
                        )
                    )
                    completed = _load_completed(output_csv) if resume else set()

                    for _, m_train, seed in train_files:
                        if (m_train, seed) in completed:
                            continue

                        tasks.append(
                            Task(
                                framework=framework,
                                dataset_dir=str(dataset_dir),
                                target=target,
                                model_path=model_config.path,
                                model_name=model_config.name,
                                train_config_path=train_config_path,
                                train_file=f"train_m{m_train}_seed{seed}.npz",
                                m_train=m_train,
                                seed=seed,
                                output_csv=output_csv,
                                problem=experiment.name,
                            )
                        )

    return tasks


def _find_train_files(dataset_dir: Path) -> list[tuple[Path, int, int]]:
    """Find and sort train_m*_seed*.npz files in a dataset directory."""
    matches = []

    for path in dataset_dir.glob("train_m*_seed*.npz"):
        match = TRAIN_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        matches.append((path, int(match.group("m")), int(match.group("seed"))))

    return sorted(matches, key=lambda item: (item[1], item[2]))


def _load_completed(output_path: str) -> set[tuple[int, int]]:
    """Return the (m_train, seed) pairs already present in a metrics CSV."""
    path = Path(output_path)
    if not path.exists():
        return set()

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return {(int(row["m_train"]), int(row["seed"])) for row in reader}


def _worker_init(blas_threads: str) -> None:
    """Pin BLAS/threading env vars once per worker process, before any
    training-library import happens in that process."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = blas_threads


def _run_pool(tasks: list[Task], workers: int, blas_threads: str) -> None:
    """Submit every task to a process pool and append results as they land."""
    total = len(tasks)
    completed_count = 0
    failed_count = 0
    start_time = time.monotonic()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(blas_threads,),
    ) as pool:
        futures = {pool.submit(_run_task, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            completed_count += 1
            elapsed = time.monotonic() - start_time

            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                print(
                    f"[{completed_count}/{total}] FAILED "
                    f"{task.problem}/{task.target}/{task.framework}/"
                    f"{task.model_name} m={task.m_train} seed={task.seed}: {exc}"
                )
                continue

            _append_row(task.output_csv, row)
            print(
                f"[{completed_count}/{total}] ({elapsed:6.0f}s elapsed) "
                f"{task.problem}/{task.target}/{task.framework}/"
                f"{task.model_name} m={task.m_train} seed={task.seed} "
                f"-> error={row['relative_test_error']:.4e} "
                f"time={row['training_time_sec']:.1f}s"
            )

    print(
        f"\nDone: {completed_count - failed_count} succeeded, "
        f"{failed_count} failed, out of {total} tasks."
    )


def _run_task(task: Task) -> dict[str, float | str | int]:
    """Run one training task in a worker process and return its metrics row."""
    from ol_reproduction.config.load import load_yaml

    model_config = load_yaml(task.model_path)
    train_config = load_yaml(task.train_config_path)

    if task.framework == "pytorch":
        import torch

        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))

        from ol_reproduction.training.pytorch_train import train_pytorch_from_files

        metrics = train_pytorch_from_files(
            dataset_dir=task.dataset_dir,
            train_file=task.train_file,
            test_file="test.npz",
            target=task.target,
            model_config_path=task.model_path,
            train_config_path=task.train_config_path,
            trial_seed=task.seed,
        )
    else:
        from ol_reproduction.training.jax_train import train_jax_from_files

        metrics = train_jax_from_files(
            dataset_dir=task.dataset_dir,
            train_file=task.train_file,
            test_file="test.npz",
            target=task.target,
            model_config_path=task.model_path,
            train_config_path=task.train_config_path,
            trial_seed=task.seed,
        )

    return {
        "problem": task.problem,
        "target": task.target,
        "framework": train_config["training"]["framework"],
        "model": model_config["model"]["name"],
        "activation": model_config["model"]["activation"],
        "depth": model_config["model"]["depth"],
        "width": model_config["model"]["width"],
        "m_train": task.m_train,
        "seed": task.seed,
        "final_train_loss": metrics["final_train_loss"],
        "relative_test_error": metrics["relative_test_error"],
        "training_time_sec": metrics["training_time_sec"],
    }


def _append_row(output_csv: str, row: dict[str, float | str | int]) -> None:
    """Append one metrics row to its CSV (single-writer, called from the
    main process only, so concurrent workers never write the same file)."""
    from ol_reproduction.evaluation.metrics_io import append_metrics_row

    append_metrics_row(path=output_csv, row=row)


if __name__ == "__main__":
    main()
