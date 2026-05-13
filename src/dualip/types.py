from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch


@dataclass
class SolverArgs:
    max_iter: int = 10000
    initial_step_size: float = 1e-5
    gamma: float = 1e-3
    max_step_size: float = 0.1
    initial_dual_path: Optional[str] = None
    gamma_decay_type: Optional[Literal["step", "interval"]] = None
    gamma_decay_params: Optional[dict] = None
    save_primal: bool = False


@dataclass
class ComputeArgs:
    host_device: str
    compute_device_num: int = 1


@dataclass
class ObjectiveArgs:
    objective_type: Literal["miplib2017", "matching"]
    use_jacobi_precondition: bool = False
    objective_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ObjectiveResult:
    dual_gradient: torch.Tensor
    dual_objective: torch.Tensor
    reg_penalty: Optional[torch.Tensor] = None
    primal_objective: Optional[torch.Tensor] = None
    primal_var: Optional[torch.Tensor] = None
    dual_val_times_grad: Optional[torch.Tensor] = None
    max_pos_slack: Optional[torch.Tensor] = None
    sum_pos_slack: Optional[torch.Tensor] = None


@dataclass
class SolverResult:
    dual_val: torch.Tensor
    dual_objective: float
    objective_result: ObjectiveResult
    dual_objective_log: list[float]
    step_size_log: list[float]
