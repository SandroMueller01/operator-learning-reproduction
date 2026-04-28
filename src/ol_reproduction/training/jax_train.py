"""JAX training loop for the reproduction experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ol_reproduction.common.config import load_yaml
from ol_reproduction.data.dataset_io import load_npz_dataset
from ol_reproduction.evaluation.relative_error import relative_l2_error
from ol_reproduction.models.jax_mlp import (
    JaxMlpConfig,
    Params,
    apply_jax_mlp,
    initialize_jax_mlp,
)


ConfigDict = dict[str, Any]
OptState = optax.OptState


def train_jax_from_files(
    dataset_dir: str | Path,
    train_file: str,
    test_file: str,
    model_config_path: str | Path,
    train_config_path: str | Path,
    target: str = "u",
) -> dict[str, float]:
    """Train a JAX MLP from saved NPZ datasets.

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
        Target variable name, for example ``u``, ``p`` or ``phi``.

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

    return train_jax(
        x_train=train_data["x"],
        y_train=train_data[target_key],
        x_test=test_data["x"],
        y_test=test_data[target_key],
        model_config=model_config,
        train_config=train_config,
    )


def train_jax(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_config: ConfigDict,
    train_config: ConfigDict,
) -> dict[str, float]:
    """Train a JAX MLP and evaluate relative test error."""
    _validate_arrays(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    x_train_jax = jnp.asarray(x_train, dtype=jnp.float32)
    y_train_jax = jnp.asarray(y_train, dtype=jnp.float32)
    x_test_jax = jnp.asarray(x_test, dtype=jnp.float32)

    input_dim = int(x_train.shape[1])
    output_dim = int(y_train.shape[1])

    model_info = model_config["model"]
    mlp_config = JaxMlpConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        depth=int(model_info["depth"]),
        width=int(model_info["width"]),
        activation=str(model_info["activation"]),
    )

    seed = int(train_config.get("reproducibility", {}).get("seed", 0))
    key = jax.random.PRNGKey(seed)
    params = initialize_jax_mlp(config=mlp_config, key=key)

    optimizer_info = train_config["training"]["optimizer"]
    optimizer = optax.adam(
        learning_rate=float(optimizer_info["learning_rate"])
    )
    opt_state = optimizer.init(params)

    epochs = int(train_config["training"]["epochs"])
    log_every = int(train_config.get("logging", {}).get("log_every", 100))

    train_step = _build_train_step(
        optimizer=optimizer,
        activation=mlp_config.activation,
    )

    start_time = time.perf_counter()

    params, opt_state, final_loss = _run_training_loop(
        params=params,
        opt_state=opt_state,
        train_step=train_step,
        x_train=x_train_jax,
        y_train=y_train_jax,
        epochs=epochs,
        log_every=log_every,
    )

    jax.block_until_ready(final_loss)
    training_time = time.perf_counter() - start_time

    y_pred = apply_jax_mlp(
        params=params,
        inputs=x_test_jax,
        activation=mlp_config.activation,
    )
    y_pred_np = np.asarray(y_pred)

    test_error = relative_l2_error(
        y_true=y_test,
        y_pred=y_pred_np,
    )

    return {
        "final_train_loss": float(final_loss),
        "relative_test_error": float(test_error),
        "training_time_sec": float(training_time),
    }


def _target_to_dataset_key(target: str) -> str:
    """Convert a target name to a dataset key."""
    normalized_target = target.strip().lower()

    if not normalized_target:
        raise ValueError("target must not be empty.")

    return f"y_{normalized_target}"


def _validate_target_key(
    data: dict[str, np.ndarray],
    target_key: str,
    split_name: str,
) -> None:
    """Validate that a dataset contains the requested target key."""
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


def _build_train_step(
    optimizer: optax.GradientTransformation,
    activation: str,
):
    """Build a JIT-compiled JAX training step."""

    @jax.jit
    def train_step(
        params: Params,
        opt_state: OptState,
        x_batch: jax.Array,
        y_batch: jax.Array,
    ) -> tuple[Params, OptState, jax.Array]:
        """Run one optimization step."""
        loss, gradients = jax.value_and_grad(_mse_loss)(
            params,
            x_batch,
            y_batch,
            activation,
        )
        updates, opt_state = optimizer.update(
            gradients,
            opt_state,
            params,
        )
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    return train_step


def _run_training_loop(
    params: Params,
    opt_state: OptState,
    train_step,
    x_train: jax.Array,
    y_train: jax.Array,
    epochs: int,
    log_every: int,
) -> tuple[Params, OptState, jax.Array]:
    """Run full-batch JAX training."""
    final_loss = jnp.asarray(jnp.nan)

    for epoch in range(1, epochs + 1):
        params, opt_state, final_loss = train_step(
            params,
            opt_state,
            x_train,
            y_train,
        )

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            loss_value = float(final_loss)
            print(f"epoch={epoch:05d} train_loss={loss_value:.6e}")

    return params, opt_state, final_loss


def _mse_loss(
    params: Params,
    inputs: jax.Array,
    targets: jax.Array,
    activation: str,
) -> jax.Array:
    """Compute mean squared error loss."""
    predictions = apply_jax_mlp(
        params=params,
        inputs=inputs,
        activation=activation,
    )
    return jnp.mean((predictions - targets) ** 2)