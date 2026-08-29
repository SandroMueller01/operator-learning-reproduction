"""Tests for the JAX training loop's checkpoint/restore and
loss-tolerance logic (paper Appendix A.2(iv)) -- mirrors
tests/test_pytorch_train.py for the JAX trainer.

Uses a scripted train_step that returns a prescribed loss sequence and a
distinguishable "params" value (the 1-indexed epoch number) per call, so
the checkpoint bookkeeping can be verified exactly.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from ol_reproduction.training.jax_train import _run_training_loop


def _make_scripted_train_step(losses: list[float]):
    state = {"call_index": 0}

    def train_step(params, opt_state, x_train, y_train):
        index = state["call_index"]
        state["call_index"] += 1
        new_params = index + 1  # marks which epoch this update happened at
        return new_params, opt_state, jnp.asarray(losses[index])

    return train_step


def _run_scripted_training(losses: list[float], loss_tolerance: float = 0.0):
    train_step = _make_scripted_train_step(losses)
    x_train = jnp.zeros((2, 1))
    y_train = jnp.zeros((2, 1))

    params, opt_state, result = _run_training_loop(
        params=0,
        opt_state=None,
        train_step=train_step,
        x_train=x_train,
        y_train=y_train,
        epochs=len(losses),
        log_every=1000,
        loss_tolerance=loss_tolerance,
    )
    return params, result


def test_smooth_decreasing_loss_never_triggers_restore() -> None:
    losses = [5.0, 4.0, 3.0, 2.0, 1.0]

    params, result = _run_scripted_training(losses)

    assert result["restored_from_checkpoint"] is False
    assert result["final_loss"] == pytest.approx(1.0)
    assert result["best_loss"] == pytest.approx(1.0)
    assert params == 5  # final epoch's own params, not a rollback


def test_late_loss_spike_triggers_restore_to_best_checkpoint() -> None:
    losses = [5.0, 3.0, 1.0, 4.0, 6.0]

    params, result = _run_scripted_training(losses)

    assert result["restored_from_checkpoint"] is True
    assert result["final_loss"] == pytest.approx(1.0)
    assert result["best_loss"] == pytest.approx(1.0)
    assert params == 3  # rolled back to the best-loss epoch


def test_tied_loss_does_not_move_the_checkpoint() -> None:
    """Strict '<' (not '<='), matching the authors' callback -- no separate
    ratio trigger. See test_pytorch_train.py's version of this test for
    the full rationale."""
    losses = [8.0, 8.0, 0.9, 5.0, 5.0]

    params, result = _run_scripted_training(losses)

    assert result["restored_from_checkpoint"] is True
    assert result["final_loss"] == pytest.approx(0.9, rel=1e-6)
    assert params == 3


def test_loss_tolerance_stops_training_early() -> None:
    losses = [1.0, 0.5, 1.0e-8, 1.0, 1.0]

    params, result = _run_scripted_training(losses, loss_tolerance=5.0e-7)

    assert result["epochs_ran"] == 3
    assert result["final_loss"] == pytest.approx(1.0e-8, rel=1e-4)
