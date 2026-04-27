import warnings
from typing import Callable

from dualip.objectives.base import BaseObjective
from dualip.utils.mlflow_utils import log_metrics


def _interval_schedule(itr: int, gamma: float, p: dict) -> float:
    """Piecewise-constant schedule: gammas[i] is held for intervals[i] iterations.

    Example: intervals=[100, 100, 500], gammas=[0.1, 0.1, 0.01]
        itr in   1..100 -> 0.1
        itr in 101..200 -> 0.1
        itr in 201..700 -> 0.01
        itr > 700       -> 0.01 (last value held)
    """
    intervals = p["intervals"]
    gammas = p["gammas"]
    if len(intervals) != len(gammas):
        raise ValueError(
            f"'interval' schedule requires intervals and gammas of equal length, "
            f"got {len(intervals)} and {len(gammas)}"
        )
    cumulative = 0
    new_gamma = gammas[-1]
    for length, g in zip(intervals, gammas):
        cumulative += length
        if itr <= cumulative:
            new_gamma = g
            break
    if new_gamma > gamma:
        warnings.warn(
            f"'interval' schedule increased gamma from {gamma} to {new_gamma} at itr={itr}; "
            f"gamma schedules are typically non-increasing.",
            stacklevel=2,
        )
    return new_gamma


# Maps decay_type -> fn(itr, gamma, params) -> new_gamma
_SCHEDULES: dict[str, Callable[[int, float, dict], float]] = {
    "step": lambda itr, gamma, p: (gamma * p["decay_factor"] if itr % p["decay_steps"] == 0 else gamma),
    "interval": _interval_schedule,
}

_REQUIRED_PARAMS: dict[str, list[str]] = {
    "step": ["decay_steps", "decay_factor"],
    "interval": ["intervals", "gammas"],
}


class GammaScheduler:
    """
    Drives gamma decay on the objective each optimizer iteration.

    To add a new schedule type, register it in _SCHEDULES:
        _SCHEDULES["my_type"] = lambda itr, gamma, params: new_gamma
    """

    def __init__(
        self,
        objective: BaseObjective,
        initial_gamma: float,
        decay_type: str,
        decay_params: dict,
    ):
        if decay_type not in _SCHEDULES:
            raise ValueError(f"Unsupported gamma decay type: {decay_type}")
        required = _REQUIRED_PARAMS.get(decay_type, [])
        missing = [k for k in required if k not in decay_params]
        if missing:
            raise ValueError(f"decay_params missing required keys for '{decay_type}': {missing}")
        self.objective = objective
        self.gamma = initial_gamma
        self.decay_params = decay_params
        self._schedule_fn = _SCHEDULES[decay_type]

    def step(self, itr: int) -> None:
        """Called after each optimizer iteration. Updates gamma on the objective if a decay fires."""
        new_gamma = self._schedule_fn(itr, self.gamma, self.decay_params)
        if new_gamma != self.gamma:
            self.gamma = new_gamma
            self.objective.set_gamma(new_gamma)
        log_metrics({"gamma": self.gamma}, step=itr)
