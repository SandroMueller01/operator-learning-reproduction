"""PyTorch training loop for the reproduction experiments."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ol_reproduction.config.load import load_yaml
from ol_reproduction.data.dataset_io import load_npz_dataset
from ol_reproduction.evaluation.relative_error import (
    relative_l2_error,
    relative_l2_error_mass_weighted,
)
from ol_reproduction.models.pytorch_mlp import (
    PyTorchMLP,
    initialize_weights,
)
from ol_reproduction.pde.mass_matrix import load_mass_matrix_npz


ConfigDict = dict[str, Any]


def train_pytorch_from_files(
    dataset_dir: str | Path,
    train_file: str,
    test_file: str,
    model_config_path: str | Path,
    train_config_path: str | Path,
    target: str = "u",
    trial_seed: int | None = None,
) -> dict[str, float]:
    """Train a PyTorch MLP from saved NPZ datasets.

    Parameters
    ----------
    dataset_dir:
        Directory containing training and testing NPZ files.
    train_file:
        Training NPZ file name.
    test_file:
        Testing NPZ file name.
    model_config_path:
        Path to model YAML config.
    train_config_path:
        Path to training YAML config.
    target:
        Target variable name. Examples are ``u``, ``p`` and ``phi``.
    trial_seed:
        Trial index (paper: 0-11), used to seed weight initialization so
        different trials get distinct, reproducible starting points. See
        ``ol_reproduction.models.pytorch_mlp.initialize_weights``.

    Returns
    -------
    dict[str, float]
        Training summary metrics.
    """
    dataset_path = Path(dataset_dir)

    train_data = load_npz_dataset(dataset_path / train_file)
    test_data = load_npz_dataset(dataset_path / test_file)

    target_key = _target_to_dataset_key(target)
    _validate_target_key(
        data=train_data,
        target_key=target_key,
        split_name="train",
    )
    _validate_target_key(
        data=test_data,
        target_key=target_key,
        split_name="test",
    )

    model_config = load_yaml(model_config_path)
    train_config = load_yaml(train_config_path)

    mass_matrix = _load_mass_matrix_for_target(dataset_path, target)
    test_weights = test_data.get("w")

    return train_pytorch(
        x_train=train_data["x"],
        y_train=train_data[target_key],
        x_test=test_data["x"],
        y_test=test_data[target_key],
        model_config=model_config,
        train_config=train_config,
        trial_seed=trial_seed,
        mass_matrix=mass_matrix,
        test_weights=test_weights,
    )


def _load_mass_matrix_for_target(dataset_path: Path, target: str):
    """Load the FEM mass matrix for ``target``'s function space, if a
    sidecar mass-matrix file exists next to the dataset.

    Diffusion cases store a single ``mass_matrix.npz`` (target is always
    ``u``); NSB cases store ``mass_matrix_u.npz`` and ``mass_matrix_p.npz``
    separately. Returns ``None`` (rather than raising) if no matching file
    is found, so callers that pass raw arrays directly (e.g. tests) are
    unaffected -- ``train_pytorch`` falls back to an unweighted L2 error
    in that case.
    """
    candidates = [
        dataset_path / f"mass_matrix_{target.strip().lower()}.npz",
        dataset_path / "mass_matrix.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_mass_matrix_npz(candidate)
    return None


def train_pytorch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_config: ConfigDict,
    train_config: ConfigDict,
    trial_seed: int | None = None,
    mass_matrix=None,
    test_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Train a PyTorch MLP and evaluate relative test error.

    Parameters
    ----------
    x_train:
        Training inputs with shape ``(m_train, input_dim)``.
    y_train:
        Training outputs with shape ``(m_train, output_dim)``.
    x_test:
        Test inputs with shape ``(m_test, input_dim)``.
    y_test:
        Test outputs with shape ``(m_test, output_dim)``.
    model_config:
        Model configuration.
    train_config:
        Training configuration.
    trial_seed:
        Trial index used to seed weight initialization (see
        ``train_pytorch_from_files``).
    mass_matrix:
        Optional FEM mass matrix (``scipy.sparse``) for the target's
        function space. When given, the test error is the paper's actual
        mass-matrix-weighted Bochner (Y-norm) error
        (``relative_l2_error_mass_weighted``); when omitted, it falls back
        to a plain unweighted relative L2 error over raw DOF values.
    test_weights:
        Optional sparse-grid quadrature weights, shape ``(m_test,)``,
        applied alongside ``mass_matrix`` per the paper's test-error
        formula (Appendix A.2(vi)). Ignored if ``mass_matrix`` is ``None``.

    Returns
    -------
    dict[str, float]
        Summary metrics containing final training loss, relative test error,
        training time and executed epochs.
    """
    _validate_arrays(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    device = _get_device(train_config["training"].get("device", "auto"))

    x_train_tensor = torch.as_tensor(
        x_train,
        dtype=torch.float32,
        device=device,
    )
    y_train_tensor = torch.as_tensor(
        y_train,
        dtype=torch.float32,
        device=device,
    )
    x_test_tensor = torch.as_tensor(
        x_test,
        dtype=torch.float32,
        device=device,
    )

    model = _build_model(
        x_train=x_train,
        y_train=y_train,
        model_config=model_config,
        device=device,
        trial_seed=trial_seed,
    )

    optimizer = _build_optimizer(
        model=model,
        train_config=train_config,
    )
    scheduler = _build_scheduler(
        optimizer=optimizer,
        train_config=train_config,
    )

    loss_function = nn.MSELoss()
    epochs = int(train_config["training"]["epochs"])
    log_every = int(train_config.get("logging", {}).get("log_every", 100))
    early_stopping_config = _get_early_stopping_config(train_config)
    # Paper: train up to 60,000 epochs or until the loss reaches this
    # tolerance, whichever comes first. Defaults to 0.0 (never trips, since
    # MSE loss is nonnegative) so configs that omit it behave exactly as
    # before this option existed.
    loss_tolerance = float(train_config["training"].get("loss_tolerance", 0.0))

    start_time = time.perf_counter()

    training_result = _run_training_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        x_train=x_train_tensor,
        y_train=y_train_tensor,
        epochs=epochs,
        log_every=log_every,
        early_stopping_config=early_stopping_config,
        loss_tolerance=loss_tolerance,
    )

    training_time = time.perf_counter() - start_time

    y_pred = _predict_numpy(
        model=model,
        inputs=x_test_tensor,
    )

    if mass_matrix is not None:
        test_error = relative_l2_error_mass_weighted(
            y_true=y_test,
            y_pred=y_pred,
            mass_matrix=mass_matrix,
            parametric_weights=test_weights,
        )
    else:
        test_error = relative_l2_error(
            y_true=y_test,
            y_pred=y_pred,
        )

    return {
        "final_train_loss": float(training_result["final_loss"]),
        "best_train_loss": float(training_result["best_loss"]),
        "relative_test_error": float(test_error),
        "training_time_sec": float(training_time),
        "epochs_ran": float(training_result["epochs_ran"]),
        "early_stopped": float(training_result["early_stopped"]),
        "restored_from_checkpoint": float(training_result["restored_from_checkpoint"]),
    }


def _target_to_dataset_key(target: str) -> str:
    """Convert a target name to the corresponding dataset key.

    Parameters
    ----------
    target:
        Target name, for example ``u``, ``p`` or ``phi``.

    Returns
    -------
    str
        Dataset key, for example ``y_u``.
    """
    normalized_target = target.strip().lower()

    if not normalized_target:
        raise ValueError("target must not be empty.")

    return f"y_{normalized_target}"


def _validate_target_key(
    data: dict[str, np.ndarray],
    target_key: str,
    split_name: str,
) -> None:
    """Validate that a dataset contains the requested target key.

    Parameters
    ----------
    data:
        Dataset dictionary.
    target_key:
        Required target key.
    split_name:
        Name of the dataset split, used in error messages.
    """
    if "x" not in data:
        raise ValueError(
            f"Dataset split {split_name!r} is missing required key 'x'. "
            f"Available keys: {sorted(data.keys())}"
        )

    if target_key not in data:
        raise ValueError(
            f"Dataset split {split_name!r} is missing target key "
            f"{target_key!r}. Available keys: {sorted(data.keys())}"
        )


def _validate_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Validate train/test arrays.

    Parameters
    ----------
    x_train:
        Training inputs.
    y_train:
        Training targets.
    x_test:
        Test inputs.
    y_test:
        Test targets.
    """
    if x_train.ndim != 2:
        raise ValueError("x_train must be two-dimensional.")

    if y_train.ndim != 2:
        raise ValueError("y_train must be two-dimensional.")

    if x_test.ndim != 2:
        raise ValueError("x_test must be two-dimensional.")

    if y_test.ndim != 2:
        raise ValueError("y_test must be two-dimensional.")

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have same sample count.")

    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("x_test and y_test must have same sample count.")

    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("x_train and x_test must have same input dimension.")

    if y_train.shape[1] != y_test.shape[1]:
        raise ValueError("y_train and y_test must have same output dimension.")


def _build_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_config: ConfigDict,
    device: torch.device,
    trial_seed: int | None = None,
) -> PyTorchMLP:
    """Build and initialize a PyTorch MLP.

    Parameters
    ----------
    x_train:
        Training input array.
    y_train:
        Training target array.
    model_config:
        Model configuration.
    device:
        PyTorch device.
    trial_seed:
        Trial index used to seed weight initialization.

    Returns
    -------
    PyTorchMLP
        Initialized model.
    """
    model_info = model_config["model"]

    model = PyTorchMLP(
        input_dim=int(x_train.shape[1]),
        output_dim=int(y_train.shape[1]),
        depth=int(model_info["depth"]),
        width=int(model_info["width"]),
        activation=str(model_info["activation"]),
    ).to(device)

    initialization_name = str(
        model_config.get("initialization", {}).get("name", "default")
    )

    initialize_weights(
        model=model,
        initialization_name=initialization_name,
        activation=str(model_info["activation"]),
        trial_seed=trial_seed,
    )

    return model


def _build_optimizer(
    model: nn.Module,
    train_config: ConfigDict,
) -> torch.optim.Optimizer:
    """Build the PyTorch optimizer.

    Parameters
    ----------
    model:
        PyTorch model.
    train_config:
        Training configuration.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer.

    Raises
    ------
    ValueError
        If the optimizer is unsupported.
    """
    optimizer_info = train_config["training"]["optimizer"]
    optimizer_name = str(optimizer_info["name"]).lower()

    learning_rate = float(optimizer_info["learning_rate"])
    weight_decay = float(optimizer_info.get("weight_decay", 0.0))

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: ConfigDict,
):
    """Build an optional PyTorch learning-rate scheduler.

    Parameters
    ----------
    optimizer:
        PyTorch optimizer.
    train_config:
        Training configuration.

    Returns
    -------
    object | None
        Scheduler object or ``None``.

    Raises
    ------
    ValueError
        If the scheduler is unsupported.
    """
    scheduler_info = train_config["training"].get("scheduler", {})
    scheduler_name = str(scheduler_info.get("name", "none")).lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "exponential_decay":
        decay_rate = float(scheduler_info["decay_rate"])
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=decay_rate,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _get_early_stopping_config(train_config: ConfigDict) -> ConfigDict:
    """Extract early stopping configuration.

    Parameters
    ----------
    train_config:
        Training configuration.

    Returns
    -------
    dict[str, Any]
        Early stopping configuration.
    """
    return train_config["training"].get(
        "early_stopping",
        {
            "enabled": False,
            "patience": 0,
            "min_delta": 0.0,
        },
    )


def _run_training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_function: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    log_every: int,
    early_stopping_config: ConfigDict,
    loss_tolerance: float = 0.0,
) -> dict[str, float | int | bool]:
    """Run full-batch PyTorch training.

    Implements the paper's actual checkpoint/restore rule, read directly
    from the authors' own ``EarlyStoppingPredictHistory`` Keras callback
    (``PDE_DATA/CODE_P/callbacks.py``): snapshot the weights every epoch
    whose loss is the best seen so far (a single trigger -- there is no
    separate loss-ratio trigger for checkpointing; a ``1/16`` loss-ratio-or
    -10000-epoch condition exists in the authors' code, but it only gates
    how often they compute/log an expensive test-set error during training,
    and has no effect on which weights get saved or restored). After
    training, restore the saved checkpoint if the final loss is worse than
    the checkpoint's loss (``current_loss > best_loss`` in their code).
    This runs unconditionally (it is the paper's core training procedure)
    and is independent of ``early_stopping_config``, which is a separate,
    off-by-default patience mechanism that halts training early rather
    than rolling back to a better checkpoint.

    Parameters
    ----------
    model:
        PyTorch model.
    optimizer:
        Optimizer.
    scheduler:
        Optional learning-rate scheduler.
    loss_function:
        Loss function.
    x_train:
        Training inputs.
    y_train:
        Training targets.
    epochs:
        Maximum number of epochs.
    log_every:
        Logging frequency.
    early_stopping_config:
        Early stopping configuration.
    loss_tolerance:
        Stop training early once the loss drops below this value (paper:
        5e-7). Defaults to 0.0, which never trips for a nonnegative MSE
        loss.

    Returns
    -------
    dict[str, float | int | bool]
        Training result.
    """
    final_loss = float("nan")
    best_loss = float("inf")
    epochs_without_improvement = 0
    epochs_ran = 0
    early_stopped = False

    early_stopping_enabled = bool(
        early_stopping_config.get("enabled", False)
    )
    patience = int(early_stopping_config.get("patience", 0))
    min_delta = float(early_stopping_config.get("min_delta", 0.0))

    checkpoint_state: dict[str, torch.Tensor] | None = None
    checkpoint_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        prediction = model(x_train)
        loss = loss_function(prediction, y_train)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        final_loss = float(loss.detach().cpu().item())
        epochs_ran = epoch

        is_best_so_far = final_loss < best_loss
        if is_best_so_far:
            best_loss = final_loss
            checkpoint_state = copy.deepcopy(model.state_dict())
            checkpoint_loss = final_loss

        if _should_log_epoch(
            epoch=epoch,
            epochs=epochs,
            log_every=log_every,
        ):
            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch:05d} "
                f"train_loss={final_loss:.6e} "
                f"lr={learning_rate:.6e}"
            )

        if early_stopping_enabled:
            improved = final_loss < best_loss - min_delta

            if improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience > 0 and epochs_without_improvement >= patience:
                early_stopped = True
                print(
                    f"Early stopping at epoch={epoch:05d} "
                    f"best_loss={best_loss:.6e}"
                )
                break

        if final_loss < loss_tolerance:
            print(
                f"Loss tolerance reached at epoch={epoch:05d} "
                f"loss={final_loss:.6e} tolerance={loss_tolerance:.6e}"
            )
            break

    restored_from_checkpoint = False
    if checkpoint_state is not None and final_loss > checkpoint_loss:
        model.load_state_dict(checkpoint_state)
        final_loss = checkpoint_loss
        restored_from_checkpoint = True

    return {
        "final_loss": final_loss,
        "best_loss": best_loss,
        "epochs_ran": epochs_ran,
        "early_stopped": early_stopped,
        "restored_from_checkpoint": restored_from_checkpoint,
    }


def _should_log_epoch(
    epoch: int,
    epochs: int,
    log_every: int,
) -> bool:
    """Check whether a training epoch should be logged.

    Parameters
    ----------
    epoch:
        Current epoch.
    epochs:
        Total number of epochs.
    log_every:
        Logging frequency.

    Returns
    -------
    bool
        Whether to log the current epoch.
    """
    return epoch == 1 or epoch % log_every == 0 or epoch == epochs


def _predict_numpy(
    model: nn.Module,
    inputs: torch.Tensor,
) -> np.ndarray:
    """Evaluate model and return NumPy predictions.

    Parameters
    ----------
    model:
        PyTorch model.
    inputs:
        Input tensor.

    Returns
    -------
    np.ndarray
        Predictions as a NumPy array.
    """
    model.eval()

    with torch.no_grad():
        predictions = model(inputs)

    return predictions.detach().cpu().numpy()


def _get_device(device_name: str) -> torch.device:
    """Resolve training device.

    Parameters
    ----------
    device_name:
        Device config value. Use ``auto`` to select CUDA if available.

    Returns
    -------
    torch.device
        PyTorch device.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)