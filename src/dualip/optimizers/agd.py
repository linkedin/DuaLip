import math
from typing import Callable, Optional

import torch

from dualip.objectives.base import BaseObjective
from dualip.optimizers.agd_utils import calculate_step_size
from dualip.types import ObjectiveResult, SolverResult
from dualip.utils.mlflow_utils import log_metrics, log_objective_result


def project_on_nn_cone(y: torch.Tensor, equality_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Projects the dual variables onto the nonnegative cone.
    """
    projected = torch.maximum(y, torch.tensor(0.0, device=y.device))
    if equality_mask is not None:
        return torch.where(equality_mask, y, projected)
    else:
        return projected


def format_objective_result_summary(iteration: int, objective_result: ObjectiveResult) -> str:
    """
    Build a one-line summary string describing the objective result for a given iteration.
    """
    try:
        grad_norm_val = float(objective_result.dual_gradient.norm().item())
        grad_norm_str = f"dual_grad_norm={grad_norm_val}"
    except Exception:
        grad_norm_str = "dual_grad_norm=<unprintable>"

    def _fmt(name, val):
        try:
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    return f"{name}={val.item()}"
                else:
                    return f"{name}.shape={tuple(val.shape)}"
            return f"{name}={val}"
        except Exception:
            return f"{name}=<unprintable>"

    parts = [
        f"iter={iteration}",
        _fmt("dual_objective", objective_result.dual_objective),
        grad_norm_str,
    ]
    opt_fields = [
        ("reg_penalty", getattr(objective_result, "reg_penalty", None)),
        ("primal_objective", getattr(objective_result, "primal_objective", None)),
        ("primal_var", getattr(objective_result, "primal_var", None)),
        ("dual_val_times_grad", getattr(objective_result, "dual_val_times_grad", None)),
        ("max_pos_slack", getattr(objective_result, "max_pos_slack", None)),
        ("sum_pos_slack", getattr(objective_result, "sum_pos_slack", None)),
    ]
    for name, val in opt_fields:
        if val is not None:
            parts.append(_fmt(name, val))
    return " | ".join([p for p in parts if p is not None])


class AcceleratedGradientDescent:
    def __init__(
        self,
        max_iter: int,
        gamma: float,
        initial_step_size: float = 1e-5,
        max_step_size: float = 0.1,
        gamma_decay_type: str = None,
        gamma_decay_params: dict = {},
        save_primal: bool = False,
        iteration_callback: Optional[Callable[[int, ObjectiveResult], None]] = None,
    ):

        self.initial_step_size = initial_step_size
        self.max_step_size = max_step_size
        self.max_iter = max_iter
        self.beta_seq = self._compute_beta_seq(self.max_iter)
        self.streams = None
        self.gamma = gamma
        self.gamma_decay_type = gamma_decay_type
        self.gamma_decay_params = gamma_decay_params
        self.save_primal = save_primal
        # Default behavior: print summary line each iteration; can be overridden by passing a callback
        self.iteration_callback: Callable[[int, ObjectiveResult], None] = (
            iteration_callback if iteration_callback is not None else self._default_iteration_callback
        )

    def _compute_beta_seq(self, max_iter: int) -> torch.Tensor:
        t_seq = torch.zeros(max_iter + 2)
        beta_seq = torch.zeros(max_iter)
        for i in range(1, max_iter + 2):
            t_seq[i] = (1 + math.sqrt(1 + 4 * (t_seq[i - 1] ** 2))) / 2
        for i in range(max_iter):
            beta_seq[i] = (1 - t_seq[i + 1]) / t_seq[i + 2]
        return beta_seq

    def _update_gamma(self, itr: int, step_size: float):
        if self.gamma_decay_type == "step":
            if itr % self.gamma_decay_params["decay_steps"] == 0:
                decay_factor = self.gamma_decay_params["decay_factor"]
                self.gamma = self.gamma * decay_factor
                self.max_step_size = step_size * decay_factor
        else:
            raise ValueError(f"Unsupported gamma decay type: {self.gamma_decay_type}")

    def _default_iteration_callback(self, iteration: int, objective_result: ObjectiveResult) -> None:
        """
        Default iteration callback that prints a one-line summary.
        """
        try:
            print(format_objective_result_summary(iteration, objective_result))
        except Exception:
            # Ensure optimizer never crashes due to logging/printing
            pass

    def maximize(self, f: BaseObjective, initial_value: torch.Tensor) -> SolverResult:
        """
        Maximizes the dual-primal objective function f.
        f must provide a method:
          - f.calculate(x) returning an object with attributes:
              * dual_gradient (torch.Tensor)
              * dual_objective (float)
              * dual_val (torch.Tensor)
        Returns a tuple: (final solution, final result, dual_obj_log, step_size_log),
        where dual_obj_log is the list of dual objective values recorded at each iteration
        and step_size_log is the list of the dynamic step size.
        """
        grad_history = []
        dual_history = []
        dual_obj_log = []  # Log of dual objective values per iteration
        step_size_log = []

        # x and y for the accelerated update.
        x = initial_value.clone()
        y = initial_value.clone()
        equality_mask = f.equality_mask

        i = 1
        while i <= self.max_iter:

            gamma_params = {"gamma": self.gamma} if self.gamma is not None else {}

            if i == self.max_iter and self.save_primal:
                objective_result: ObjectiveResult = f.calculate(
                    dual_val=x, **gamma_params, save_primal=self.save_primal
                )
            else:
                objective_result: ObjectiveResult = f.calculate(dual_val=x, **gamma_params)

            # Invoke decoupled iteration callback (prints by default; can be overridden)
            self.iteration_callback(i, objective_result)

            dual_obj = objective_result.dual_objective.cpu().item()
            dual_obj_log.append(dual_obj)

            step_size = calculate_step_size(
                objective_result.dual_gradient,
                y,
                grad_history,
                dual_history,
                initial_step_size=self.initial_step_size,
                max_step_size=self.max_step_size,
            )

            step_size_log.append(step_size)
            # Gradient ascent step.
            y_new = x + objective_result.dual_gradient * step_size
            y_new = project_on_nn_cone(y_new, equality_mask)
            # Accelerated update.
            x = (y_new * (1.0 - self.beta_seq[i - 1])) + (y * self.beta_seq[i - 1])
            y = y_new
            if self.gamma is not None and self.gamma_decay_type is not None:
                self._update_gamma(i, step_size)

            # Log iteration metrics (will check MLflow state internally)
            iteration_metrics = {
                "step_size": step_size,
                "dual_objective": dual_obj,
            }

            if self.gamma is not None:
                iteration_metrics["gamma"] = self.gamma

            log_metrics(iteration_metrics, step=i)

            # Log objective result details
            log_objective_result(objective_result, step=i)

            i += 1

        solver_result = SolverResult(
            dual_val=y,
            dual_objective=dual_obj,
            objective_result=objective_result,
            dual_objective_log=dual_obj_log,
            step_size_log=step_size_log,
        )
        return solver_result
