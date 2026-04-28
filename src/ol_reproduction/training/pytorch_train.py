"""PyTorch training loop for the reproduction experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ol_reproduction.common.config import load_yaml
from ol_reproduction.data.dataset_io import load_npz_dataset
from ol_reproduction.evaluation.relative_error import relative_l2_error
from ol_reproduction.models.pytorch_mlp import (
    PyTorchMLP,
    initialize_weights,
)


ConfigDict = dict[str, Any]


def train_pytorch_from_files(
    dataset_dir: str | Path,
    train_file: str,
    test_file: str,
    model_config_path: str | Path,
    train_config_path: str | Path,
    target: str = "u",
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
        Target variable name. Supported values depend on the dataset, for
        example ``u``, ``p`` or ``phi``.

    Returns
    -------
    dict[str, float]
        Training summary metrics.
    """
    dataset_path = Path(dataset_dir)

    train_data = load_npz_dataset(dataset_path / train_file)
    test_data = load_npz_dataset(dataset_path / test_file)

    target_key = _target_to_dataset_key(target)
    _validate_target_key(train_data, target_key, split_name="train")
    _validate_target_key(test_data, target_key, split_name="test")

    model_config = load_yaml(model_config_path)
    train_config = load_yaml(train_config_path)

    return train_pytorch(
        x_train=train_data["x"],
        y_train=train_data[target_key],
        x_test=test_data["x"],
        y_test=test_data[target_key],
        model_config=model_config,
        train_config=train_config,
    )


def train_pytorch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_config: ConfigDict,
    train_config: ConfigDict,
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

    Returns
    -------
    dict[str, float]
        Summary metrics containing final training loss, relative test error,
        and training time.
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
    )

    optimizer = _build_optimizer(
        model=model,
        train_config=train_config,
    )

    loss_function = nn.MSELoss()
    epochs = int(train_config["training"]["epochs"])
    log_every = int(train_config.get("logging", {}).get("log_every", 100))

    start_time = time.perf_counter()

    final_loss = _run_training_loop(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        x_train=x_train_tensor,
        y_train=y_train_tensor,
        epochs=epochs,
        log_every=log_every,
    )

    training_time = time.perf_counter() - start_time

    y_pred = _predict_numpy(
        model=model,
        inputs=x_test_tensor,
    )

    test_error = relative_l2_error(
        y_true=y_test,
        y_pred=y_pred,
    )

    return {
        "final_train_loss": float(final_loss),
        "relative_test_error": float(test_error),
        "training_time_sec": float(training_time),
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
    """Validate train/test arrays."""
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
) -> PyTorchMLP:
    """Build and initialize a PyTorch MLP."""
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
    )

    return model


def _build_optimizer(
    model: nn.Module,
    train_config: ConfigDict,
) -> torch.optim.Optimizer:
    """Build the PyTorch optimizer."""
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


def _run_training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    log_every: int,
) -> float:
    """Run full-batch PyTorch training."""
    final_loss = float("nan")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        prediction = model(x_train)
        loss = loss_function(prediction, y_train)

        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu().item())

        if _should_log_epoch(
            epoch=epoch,
            epochs=epochs,
            log_every=log_every,
        ):
            print(f"epoch={epoch:05d} train_loss={final_loss:.6e}")

    return final_loss


def _should_log_epoch(
    epoch: int,
    epochs: int,
    log_every: int,
) -> bool:
    """Check whether a training epoch should be logged."""
    return epoch == 1 or epoch % log_every == 0 or epoch == epochs


def _predict_numpy(
    model: nn.Module,
    inputs: torch.Tensor,
) -> np.ndarray:
    """Evaluate model and return NumPy predictions."""
    model.eval()

    with torch.no_grad():
        predictions = model(inputs)

    return predictions.detach().cpu().numpy()


def _get_device(device_name: str) -> torch.device:
    """Resolve training device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)