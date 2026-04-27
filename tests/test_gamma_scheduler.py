import re

import pytest
import torch

from dualip.gamma_scheduler import GammaScheduler
from dualip.objectives.base import BaseObjective
from dualip.optimizers.agd import AcceleratedGradientDescent
from dualip.types import ObjectiveResult


class GammaTrackingObjective(BaseObjective):
    """Minimal objective that records every gamma it is given via set_gamma()."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.gamma_history: list[float] = [gamma]
        self.equality_mask = None

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma
        self.gamma_history.append(gamma)

    def calculate(self, dual_val: torch.Tensor, save_primal: bool = False, **kwargs) -> ObjectiveResult:
        grad = -2.0 * dual_val
        obj = -(dual_val**2).sum()
        return ObjectiveResult(dual_gradient=grad, dual_objective=obj)


def test_step_scheduler_calls_set_gamma():
    """GammaScheduler with 'step' decay calls set_gamma on the objective at the right iterations."""
    objective = GammaTrackingObjective(gamma=1.0)
    scheduler = GammaScheduler(
        objective=objective,
        initial_gamma=1.0,
        decay_type="step",
        decay_params={"decay_steps": 3, "decay_factor": 0.5},
    )

    for itr in range(1, 7):
        scheduler.step(itr)

    # Decay fires at itr=3 and itr=6
    assert len(objective.gamma_history) == 3  # initial + 2 decays
    assert objective.gamma_history[0] == pytest.approx(1.0)
    assert objective.gamma_history[1] == pytest.approx(0.5)
    assert objective.gamma_history[2] == pytest.approx(0.25)


def test_step_scheduler_decays_gamma_at_correct_iterations():
    """GammaScheduler decays gamma at the right iterations and leaves it unchanged otherwise."""
    objective = GammaTrackingObjective(gamma=1.0)
    scheduler = GammaScheduler(
        objective=objective,
        initial_gamma=1.0,
        decay_type="step",
        decay_params={"decay_steps": 2, "decay_factor": 0.5},
    )

    scheduler.step(1)  # no decay
    assert scheduler.gamma == pytest.approx(1.0)
    scheduler.step(2)  # decay fires
    assert scheduler.gamma == pytest.approx(0.5)


def test_agd_step_callback_drives_gamma():
    """step_callback correctly drives GammaScheduler which updates objective gamma."""
    objective = GammaTrackingObjective(gamma=1.0)
    scheduler = GammaScheduler(
        objective=objective,
        initial_gamma=1.0,
        decay_type="step",
        decay_params={"decay_steps": 5, "decay_factor": 0.5},
    )

    solver = AcceleratedGradientDescent(
        max_iter=10,
        initial_step_size=1e-3,
        max_step_size=0.1,
    )
    initial = torch.zeros(2)
    solver.maximize(objective, initial, step_callback=scheduler.step)

    # Decay fires at itr=5 and itr=10 → 2 updates on top of initial
    assert len(objective.gamma_history) == 3
    assert objective.gamma_history[1] == pytest.approx(0.5)
    assert objective.gamma_history[2] == pytest.approx(0.25)


def test_unsupported_decay_type_raises():
    objective = GammaTrackingObjective(gamma=1.0)
    with pytest.raises(ValueError, match="Unsupported gamma decay type"):
        GammaScheduler(objective=objective, initial_gamma=1.0, decay_type="none", decay_params={})


def test_missing_decay_params_raises():
    objective = GammaTrackingObjective(gamma=1.0)
    expected_message = "decay_params missing required keys for 'step': ['decay_steps', 'decay_factor']"
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        GammaScheduler(objective=objective, initial_gamma=1.0, decay_type="step", decay_params={})


def test_agd_with_gamma_scheduler_decays_gamma():
    """GammaScheduler decays objective gamma at the right iterations when wired via step_callback."""
    objective = GammaTrackingObjective(gamma=1.0)
    scheduler = GammaScheduler(
        objective=objective,
        initial_gamma=1.0,
        decay_type="step",
        decay_params={"decay_steps": 3, "decay_factor": 0.5},
    )
    solver = AcceleratedGradientDescent(
        max_iter=6,
        initial_step_size=1e-3,
        max_step_size=0.1,
    )
    solver.maximize(objective, torch.zeros(2), step_callback=scheduler.step)

    # Decay fires at itr=3 and itr=6 → gamma halved twice
    assert scheduler.gamma == pytest.approx(0.25)
    assert len(objective.gamma_history) == 3
    # solver's max_step_size is untouched
    assert solver.max_step_size == pytest.approx(0.1)


def test_interval_scheduler_piecewise_constant_gamma():
    """GammaScheduler with 'interval' decay holds gammas[i] for intervals[i] iterations."""
    objective = GammaTrackingObjective(gamma=0.1)
    scheduler = GammaScheduler(
        objective=objective,
        initial_gamma=0.1,
        decay_type="interval",
        decay_params={"intervals": [3, 2, 4], "gammas": [0.1, 0.05, 0.01]},
    )

    # itrs 1..3 -> 0.1 (no change from initial)
    for itr in range(1, 4):
        scheduler.step(itr)
    assert scheduler.gamma == pytest.approx(0.1)
    assert len(objective.gamma_history) == 1

    # itrs 4..5 -> 0.05
    scheduler.step(4)
    assert scheduler.gamma == pytest.approx(0.05)
    scheduler.step(5)
    assert scheduler.gamma == pytest.approx(0.05)
    assert len(objective.gamma_history) == 2

    # itrs 6..9 -> 0.01
    scheduler.step(6)
    assert scheduler.gamma == pytest.approx(0.01)
    for itr in range(7, 10):
        scheduler.step(itr)
    assert scheduler.gamma == pytest.approx(0.01)

    # past end -> last value held
    scheduler.step(100)
    assert scheduler.gamma == pytest.approx(0.01)
    assert objective.gamma_history == pytest.approx([0.1, 0.05, 0.01])


def test_interval_scheduler_mismatched_lengths_raises():
    objective = GammaTrackingObjective(gamma=1.0)
    scheduler = GammaScheduler(
        objective=objective,
        initial_gamma=1.0,
        decay_type="interval",
        decay_params={"intervals": [10, 20], "gammas": [0.1, 0.05, 0.01]},
    )
    with pytest.raises(ValueError, match="intervals and gammas of equal length"):
        scheduler.step(1)


def test_interval_scheduler_missing_params_raises():
    objective = GammaTrackingObjective(gamma=1.0)
    expected_message = "decay_params missing required keys for 'interval': ['intervals', 'gammas']"
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        GammaScheduler(
            objective=objective,
            initial_gamma=1.0,
            decay_type="interval",
            decay_params={},
        )


def test_interval_scheduler_warns_on_gamma_increase():
    """Warn when the schedule would raise gamma above the current value."""
    objective = GammaTrackingObjective(gamma=0.01)
    scheduler = GammaScheduler(
        objective=objective,
        initial_gamma=0.01,
        decay_type="interval",
        decay_params={"intervals": [2, 2], "gammas": [0.1, 0.05]},
    )
    with pytest.warns(UserWarning, match="increased gamma"):
        scheduler.step(1)
    assert scheduler.gamma == pytest.approx(0.1)


def test_agd_with_interval_gamma_scheduler():
    """GammaScheduler with 'interval' decay wired to AGD via step_callback."""
    objective = GammaTrackingObjective(gamma=0.1)
    scheduler = GammaScheduler(
        objective=objective,
        initial_gamma=0.1,
        decay_type="interval",
        decay_params={"intervals": [3, 3], "gammas": [0.1, 0.01]},
    )
    solver = AcceleratedGradientDescent(
        max_iter=6,
        initial_step_size=1e-3,
        max_step_size=0.1,
    )
    solver.maximize(objective, torch.zeros(2), step_callback=scheduler.step)

    # Gamma switches from 0.1 -> 0.01 at itr=4
    assert scheduler.gamma == pytest.approx(0.01)
    assert objective.gamma_history == pytest.approx([0.1, 0.01])


def test_agd_with_gamma_scheduler_no_decay_between_steps():
    """Gamma and solver max_step_size are unchanged when no decay fires."""
    objective = GammaTrackingObjective(gamma=1.0)
    scheduler = GammaScheduler(
        objective=objective,
        initial_gamma=1.0,
        decay_type="step",
        decay_params={"decay_steps": 10, "decay_factor": 0.5},
    )
    solver = AcceleratedGradientDescent(
        max_iter=5,
        initial_step_size=1e-3,
        max_step_size=0.1,
    )
    solver.maximize(objective, torch.zeros(2), step_callback=scheduler.step)

    # No decay fires within 5 iterations (decay_steps=10)
    assert scheduler.gamma == pytest.approx(1.0)
    assert solver.max_step_size == pytest.approx(0.1)
    assert len(objective.gamma_history) == 1  # only the initial gamma
