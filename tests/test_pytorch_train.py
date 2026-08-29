"""Tests for the PyTorch training loop's checkpoint/restore and
loss-tolerance logic (paper Appendix A.2(iv)).

Uses scripted model/loss stand-ins that produce a prescribed loss
trajectory and a distinguishable, deterministic parameter trajectory, so
the checkpoint bookkeeping can be verified exactly rather than depending
on real (unpredictable) gradient-descent dynamics.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ol_reproduction.training.pytorch_train import _run_training_loop


class _ScriptedParamModel(nn.Module):
    """A model whose single parameter is set to the 1-indexed epoch number
    after each step, so state_dict() uniquely identifies which epoch a
    checkpoint was taken at."""

    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self._epoch_counter = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.param.expand(inputs.shape[0], 1)

    def mark_epoch(self) -> None:
        self._epoch_counter += 1
        with torch.no_grad():
            self.param.fill_(float(self._epoch_counter))


class _ScriptedSequenceLoss(nn.Module):
    """Returns a prescribed loss value per call, in sequence, while
    remaining a valid differentiable tensor (gradient is always zero, since
    only the *bookkeeping* logic is under test here, not real learning)."""

    def __init__(self, losses: list[float]) -> None:
        super().__init__()
        self.losses = losses
        self.call_index = 0

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        value = self.losses[self.call_index]
        self.call_index += 1
        return prediction.sum() * 0.0 + value


class _NoOpOptimizer:
    """Stands in for a real optimizer: zero_grad()/step() are no-ops (the
    model's parameter trajectory is instead driven deterministically by
    _ScriptedParamModel.mark_epoch()), but exposes param_groups so the
    training loop's logging code (which reads the current LR) still works.
    """

    def __init__(self, model: _ScriptedParamModel) -> None:
        self.param_groups = [{"lr": 0.01}]
        self._model = model

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def step(self) -> None:
        self._model.mark_epoch()


def _run_scripted_training(
    losses: list[float],
    loss_tolerance: float = 0.0,
) -> tuple[dict, _ScriptedParamModel]:
    model = _ScriptedParamModel()
    optimizer = _NoOpOptimizer(model)
    loss_function = _ScriptedSequenceLoss(losses)
    x_train = torch.zeros((2, 1))
    y_train = torch.zeros((2, 1))

    result = _run_training_loop(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss_function=loss_function,
        x_train=x_train,
        y_train=y_train,
        epochs=len(losses),
        log_every=1000,
        early_stopping_config={"enabled": False, "patience": 0, "min_delta": 0.0},
        loss_tolerance=loss_tolerance,
    )
    return result, model


def test_smooth_decreasing_loss_never_triggers_restore() -> None:
    """A monotonically decreasing loss should never need a rollback: the
    final epoch's weights are already the best."""
    losses = [5.0, 4.0, 3.0, 2.0, 1.0]

    result, model = _run_scripted_training(losses)

    assert result["restored_from_checkpoint"] is False
    assert result["final_loss"] == 1.0
    assert result["best_loss"] == 1.0
    # Final param should be epoch 5 (the last epoch's own weights, not a
    # rollback to an earlier epoch).
    assert model.param.item() == 5.0


def test_late_loss_spike_triggers_restore_to_best_checkpoint() -> None:
    """A late spike (paper: nonmonotonic loss from nonconvex optimization)
    must roll back to the best checkpoint, not keep the spiked weights."""
    # Loss decreases to a minimum of 1.0 at epoch 3, then spikes upward and
    # stays worse than 1.0 for the remaining epochs.
    losses = [5.0, 3.0, 1.0, 4.0, 6.0]

    result, model = _run_scripted_training(losses)

    assert result["restored_from_checkpoint"] is True
    assert result["final_loss"] == 1.0
    assert result["best_loss"] == 1.0
    # The restored model's parameter should match epoch 3 (where the best
    # loss/checkpoint was captured), not epoch 5 (the spiked final epoch).
    assert model.param.item() == 3.0


def test_tied_loss_does_not_move_the_checkpoint() -> None:
    """The authors' own callback (``PDE_DATA/CODE_P/callbacks.py``) checks
    ``current_loss < best_loss`` (strict), not ``<=`` -- so there is no
    separate loss-ratio checkpoint trigger, only this single best-so-far
    rule (a 1/16 ratio does exist in their code, but only gates how often
    an expensive test-set error is logged, never which weights get saved).
    A repeated (tied) loss must not move the checkpoint away from the
    epoch that first achieved it.
    """
    losses = [8.0, 8.0, 0.9, 5.0, 5.0]

    result, model = _run_scripted_training(losses)

    assert result["restored_from_checkpoint"] is True
    assert result["final_loss"] == pytest.approx(0.9, rel=1e-6)
    # Best loss (0.9) occurs at epoch 3; the tie at epoch 2 (loss==8.0,
    # same as epoch 1) must not have moved the checkpoint to epoch 2.
    assert model.param.item() == 3.0


def test_loss_tolerance_stops_training_early() -> None:
    """An easy synthetic problem meeting the tolerance should halt well
    before the configured epoch cap, without needing 60,000 epochs."""
    losses = [1.0, 0.5, 1.0e-8, 1.0, 1.0]  # would keep going without tolerance

    result, model = _run_scripted_training(losses, loss_tolerance=5.0e-7)

    assert result["epochs_ran"] == 3
    assert result["final_loss"] == pytest.approx(1.0e-8, rel=1e-4)
