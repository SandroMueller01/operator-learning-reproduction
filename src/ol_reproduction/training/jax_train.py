"""JAX training loop for the reproduction experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ol_reproduction.config.load import load_yaml
from ol_reproduction.data.dataset_io import load_npz_dataset
from ol_reproduction.evaluation.relative_error import (
    relative_l2_error,
    relative_l2_error_mass_weighted,
)
from ol_reproduction.models.jax_mlp import (
    JaxMlpConfig,
    Params,
    apply_jax_mlp,
    initialize_jax_mlp,
)
from ol_reproduction.pde.mass_matrix import load_mass_matrix_npz


ConfigDict = dict[str, Any]
OptState = optax.OptState


def train_jax_from_files(
    dataset_dir: str | Path,
    train_file: str,
    test_file: str,
    model_config_path: str | Path,
    train_config_path: str | Path,
    target: str = "u",
    trial_seed: int | None = None,
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
    trial_seed:
        Trial index (paper: 0-11) used to seed the PRNG key for weight
        initialization. Overrides ``train_config["reproducibility"]["seed"]``
        when given -- without this, every trial sharing one train_config
        file would silently reuse the same fixed seed, defeating the point
        of running multiple trials per architecture/m/coefficient/dimension
        combo (see JaxMlpConfig/initialize_jax_mlp).

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

    mass_matrix = _load_mass_matrix_for_target(dataset_path, target)
    test_weights = test_data.get("w")

    return train_jax(
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
    sidecar mass-matrix file exists next to the dataset. See
    ``pytorch_train._load_mass_matrix_for_target`` (identical logic, kept
    duplicated to avoid a training-module cross-import)."""
    candidates = [
        dataset_path / f"mass_matrix_{target.strip().lower()}.npz",
        dataset_path / "mass_matrix.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_mass_matrix_npz(candidate)
    return None


def train_jax(
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
    """Train a JAX MLP and evaluate relative test error.

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
    model_config:
        Model configuration.
    train_config:
        Training configuration.
    trial_seed:
        Trial index used to seed the PRNG key (overrides
        ``train_config["reproducibility"]["seed"]`` when given).
    mass_matrix:
        Optional FEM mass matrix (``scipy.sparse``) for the target's
        function space -- see ``pytorch_train.train_pytorch`` for the full
        rationale. When omitted, falls back to a plain unweighted relative
        L2 error.
    test_weights:
        Optional sparse-grid quadrature weights, shape ``(m_test,)``.
        Ignored if ``mass_matrix`` is ``None``.

    Returns
    -------
    dict[str, float]
        Training summary metrics.
    """
    _validate_arrays(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    x_train_jax = jnp.asarray(x_train, dtype=jnp.float32)
    y_train_jax = jnp.asarray(y_train, dtype=jnp.float32)
    x_test_jax = jnp.asarray(x_test, dtype=jnp.float32)

    model_info = model_config["model"]
    initialization_info = model_config.get("initialization", {})

    mlp_config = JaxMlpConfig(
        input_dim=int(x_train.shape[1]),
        output_dim=int(y_train.shape[1]),
        depth=int(model_info["depth"]),
        width=int(model_info["width"]),
        activation=str(model_info["activation"]),
        initialization=str(initialization_info.get("name", "default")),
    )

    if trial_seed is not None:
        seed = trial_seed
    else:
        seed = int(train_config.get("reproducibility", {}).get("seed", 0))
    key = jax.random.PRNGKey(seed)
    params = initialize_jax_mlp(config=mlp_config, key=key)

    optimizer = _build_optimizer(train_config=train_config)
    opt_state = optimizer.init(params)

    epochs = int(train_config["training"]["epochs"])
    log_every = int(train_config.get("logging", {}).get("log_every", 100))
    # Paper: train up to 60,000 epochs or until the loss reaches this
    # tolerance, whichever comes first. Defaults to 0.0 (never trips, since
    # MSE loss is nonnegative) so configs that omit it are unaffected.
    loss_tolerance = float(train_config["training"].get("loss_tolerance", 0.0))

    train_step = _build_train_step(
        optimizer=optimizer,
        activation=mlp_config.activation,
    )

    start_time = time.perf_counter()

    params, opt_state, training_result = _run_training_loop(
        params=params,
        opt_state=opt_state,
        train_step=train_step,
        x_train=x_train_jax,
        y_train=y_train_jax,
        epochs=epochs,
        log_every=log_every,
        loss_tolerance=loss_tolerance,
    )

    training_time = time.perf_counter() - start_time

    y_pred = apply_jax_mlp(
        params=params,
        inputs=x_test_jax,
        activation=mlp_config.activation,
    )
    y_pred_np = np.asarray(y_pred)

    if mass_matrix is not None:
        test_error = relative_l2_error_mass_weighted(
            y_true=y_test,
            y_pred=y_pred_np,
            mass_matrix=mass_matrix,
            parametric_weights=test_weights,
        )
    else:
        test_error = relative_l2_error(
            y_true=y_test,
            y_pred=y_pred_np,
        )

    return {
        "final_train_loss": float(training_result["final_loss"]),
        "best_train_loss": float(training_result["best_loss"]),
        "relative_test_error": float(test_error),
        "training_time_sec": float(training_time),
        "epochs_ran": float(training_result["epochs_ran"]),
        "restored_from_checkpoint": float(training_result["restored_from_checkpoint"]),
    }


def _build_optimizer(
    train_config: ConfigDict,
) -> optax.GradientTransformation:
    """Build Optax optimizer.

    Parameters
    ----------
    train_config:
        Training configuration.

    Returns
    -------
    optax.GradientTransformation
        Configured optimizer.

    Raises
    ------
    ValueError
        If the optimizer is unsupported.
    """
    optimizer_info = train_config["training"]["optimizer"]
    optimizer_name = str(optimizer_info["name"]).lower()

    learning_rate = float(optimizer_info["learning_rate"])
    learning_rate_or_schedule = _build_lr_schedule(
        train_config=train_config,
        base_learning_rate=learning_rate,
    )

    if optimizer_name == "adam":
        weight_decay = float(optimizer_info.get("weight_decay", 0.0))

        if weight_decay > 0.0:
            return optax.adamw(
                learning_rate=learning_rate_or_schedule,
                weight_decay=weight_decay,
            )

        return optax.adam(learning_rate=learning_rate_or_schedule)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_lr_schedule(
    train_config: ConfigDict,
    base_learning_rate: float,
):
    """Build a JAX/Optax learning-rate value or schedule.

    Mirrors ``pytorch_train._build_scheduler``'s config shape
    (``training.scheduler.name``/``decay_rate``). Unlike PyTorch, Optax has
    no separate scheduler object -- the schedule is passed directly as the
    optimizer's ``learning_rate``. ``transition_steps=1, staircase=True``
    makes ``optax.exponential_decay`` multiply the learning rate by
    ``decay_rate`` once per training step, matching PyTorch's
    ``ExponentialLR(gamma=decay_rate)`` (which decays once per
    ``scheduler.step()`` call, i.e. once per epoch, since training here is
    full-batch).

    Returns
    -------
    float | optax.Schedule
        A plain float if no scheduler is configured (or ``"none"``), or an
        Optax schedule function otherwise.
    """
    scheduler_info = train_config["training"].get("scheduler", {})
    scheduler_name = str(scheduler_info.get("name", "none")).lower()

    if scheduler_name == "none":
        return base_learning_rate

    if scheduler_name == "exponential_decay":
        decay_rate = float(scheduler_info["decay_rate"])
        return optax.exponential_decay(
            init_value=base_learning_rate,
            transition_steps=1,
            decay_rate=decay_rate,
            staircase=True,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _target_to_dataset_key(target: str) -> str:
    """Convert a target name to a dataset key.

    Parameters
    ----------
    target:
        Target name.

    Returns
    -------
    str
        Dataset key.
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
        Dataset split name.
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


def _build_train_step(
    optimizer: optax.GradientTransformation,
    activation: str,
):
    """Build a JIT-compiled JAX training step.

    Parameters
    ----------
    optimizer:
        Optax optimizer.
    activation:
        Activation function name.

    Returns
    -------
    Callable
        JIT-compiled training step.
    """

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
    loss_tolerance: float = 0.0,
) -> tuple[Params, OptState, dict[str, float | int | bool]]:
    """Run full-batch JAX training.

    Implements the paper's actual checkpoint/restore rule (see
    ``pytorch_train._run_training_loop`` for the full derivation from the
    authors' Keras callback): snapshot ``params`` whenever the current loss
    is the best seen so far (single trigger -- there is no separate
    loss-ratio checkpoint trigger); restore the checkpoint at the end if
    the final loss is worse than it. JAX pytrees are immutable, so
    "snapshotting" is just keeping a reference to the current ``params`` --
    no deep copy is needed the way PyTorch's ``state_dict()`` requires.

    Parameters
    ----------
    params:
        Model parameters.
    opt_state:
        Optimizer state.
    train_step:
        JIT-compiled training step.
    x_train:
        Training inputs.
    y_train:
        Training targets.
    epochs:
        Number of epochs.
    log_every:
        Logging frequency.
    loss_tolerance:
        Stop training early once the loss drops below this value (paper:
        5e-7). Defaults to 0.0, which never trips for a nonnegative MSE loss.

    Returns
    -------
    tuple[Params, OptState, dict[str, float | int | bool]]
        Updated parameters, optimizer state, and a result dict with
        final_loss/best_loss/epochs_ran/restored_from_checkpoint.
    """
    final_loss = jnp.asarray(jnp.nan)
    final_loss_value = float("nan")
    best_loss = float("inf")
    epochs_ran = 0

    checkpoint_params: Params | None = None
    checkpoint_loss = float("inf")

    for epoch in range(1, epochs + 1):
        params, opt_state, final_loss = train_step(
            params,
            opt_state,
            x_train,
            y_train,
        )
        final_loss_value = float(final_loss)
        epochs_ran = epoch

        is_best_so_far = final_loss_value < best_loss
        if is_best_so_far:
            best_loss = final_loss_value
            checkpoint_params = params
            checkpoint_loss = final_loss_value

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"epoch={epoch:05d} train_loss={final_loss_value:.6e}")

        if final_loss_value < loss_tolerance:
            print(
                f"Loss tolerance reached at epoch={epoch:05d} "
                f"loss={final_loss_value:.6e} tolerance={loss_tolerance:.6e}"
            )
            break

    restored_from_checkpoint = False
    if checkpoint_params is not None and final_loss_value > checkpoint_loss:
        params = checkpoint_params
        final_loss_value = checkpoint_loss
        restored_from_checkpoint = True

    result = {
        "final_loss": final_loss_value,
        "best_loss": best_loss,
        "epochs_ran": epochs_ran,
        "restored_from_checkpoint": restored_from_checkpoint,
    }
    return params, opt_state, result


def _mse_loss(
    params: Params,
    inputs: jax.Array,
    targets: jax.Array,
    activation: str,
) -> jax.Array:
    """Compute mean squared error loss.

    Parameters
    ----------
    params:
        Model parameters.
    inputs:
        Input batch.
    targets:
        Target batch.
    activation:
        Activation function name.

    Returns
    -------
    jax.Array
        Scalar MSE loss.
    """
    predictions = apply_jax_mlp(
        params=params,
        inputs=inputs,
        activation=activation,
    )
    return jnp.mean((predictions - targets) ** 2)