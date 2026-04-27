import math
from typing import Callable, Optional

import torch
import torch.distributed as dist

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
    """
    Accelerated Gradient Descent optimizer (pure dual update).

    Gamma scheduling is handled externally via step_callback passed to maximize().
    See GammaScheduler for the built-in step / interval decay implementations.
    """

    def __init__(
        self,
        max_iter: int,
        initial_step_size: float = 1e-5,
        max_step_size: float = 0.1,
        save_primal: bool = False,
        iteration_callback: Optional[Callable[[int, ObjectiveResult], None]] = None,
    ):
        self.initial_step_size = initial_step_size
        self.max_step_size = max_step_size
        self.max_iter = max_iter
        self.beta_seq = self._compute_beta_seq(self.max_iter)
        self.streams = None
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

    def _default_iteration_callback(self, iteration: int, objective_result: ObjectiveResult) -> None:
        """
        Default iteration callback that prints a one-line summary.
        """
        try:
            print(format_objective_result_summary(iteration, objective_result))
        except Exception:
            # Ensure optimizer never crashes due to logging/printing
            pass

    def maximize(
        self,
        f: BaseObjective,
        initial_value: torch.Tensor,
        rank: int = 0,
        step_callback: Optional[Callable[[int], None]] = None,
    ) -> SolverResult:
        """
        Maximizes the dual objective f.

        Args:
            f: objective implementing BaseObjective.calculate(dual_val, save_primal).
                Objectives that use a regularization parameter own it internally;
                update it externally via f.set_gamma().
            initial_value: starting dual variable
            rank: distributed rank (0 = primary)
            step_callback: optional callable(itr) -> None, called after each iteration.
                Use this to drive gamma scheduling via GammaScheduler.

        Returns a SolverResult with the final dual / objective and per-iteration logs.
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

            # ALL ranks participate in calculate (for distributed objectives)
            if i == self.max_iter and self.save_primal:
                objective_result: ObjectiveResult = f.calculate(dual_val=x, save_primal=self.save_primal, rank=rank)
            else:
                objective_result: ObjectiveResult = f.calculate(dual_val=x, rank=rank)

            # Only rank 0 performs optimizer updates
            if rank == 0:
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

                # Drive external scheduling (e.g. gamma decay via GammaScheduler)
                if step_callback is not None:
                    step_callback(i)

                # Log iteration metrics (will check MLflow state internally)
                iteration_metrics = {
                    "step_size": step_size,
                    "dual_objective": dual_obj,
                }

                log_metrics(iteration_metrics, step=i)

                # Log objective result details
                log_objective_result(objective_result, step=i)

            # Broadcast updated x and y from rank 0 to all other ranks
            if dist.is_initialized():
                dist.broadcast(x, src=0)
                dist.broadcast(y, src=0)

            i += 1

        # Only rank 0 returns meaningful result
        if rank == 0:
            solver_result = SolverResult(
                dual_val=y,
                dual_objective=dual_obj,
                objective_result=objective_result,
                dual_objective_log=dual_obj_log,
                step_size_log=step_size_log,
            )
        else:
            # Non-zero ranks return a minimal result
            solver_result = SolverResult(
                dual_val=y,
                dual_objective=0.0,
                objective_result=objective_result,
                dual_objective_log=[],
                step_size_log=[],
            )

        return solver_result
