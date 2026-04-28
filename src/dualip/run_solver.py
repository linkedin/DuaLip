from dataclasses import fields

import torch

from dualip.objectives.base import BaseInputArgs
from dualip.objectives.matching import MatchingSolverDualObjectiveFunction
from dualip.objectives.miplib import MIPLIB2017ObjectiveFunction
from dualip.optimizers.agd import AcceleratedGradientDescent, SolverResult
from dualip.types import ComputeArgs, ObjectiveArgs, SolverArgs


def transfer_tensors_to_device(input_args: BaseInputArgs, device: str) -> BaseInputArgs:
    """Return a copy of ``input_args`` with all tensor fields moved to ``device``."""
    field_values = {}
    for field in fields(input_args):
        value = getattr(input_args, field.name)
        field_values[field.name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return type(input_args)(**field_values)


def build_objective(
    input_args: BaseInputArgs,
    solver_args: SolverArgs,
    compute_args: ComputeArgs,
    objective_args: ObjectiveArgs,
):
    if compute_args.compute_device_num != 1:
        raise NotImplementedError(
            "run_solver currently supports compute_device_num=1 only. "
            "For multi-GPU / multi-node usage, launch with torchrun and construct "
            "MatchingSolverDualObjectiveFunctionDistributed directly on each rank "
            "with its local data partition. See tests/distributed/test_matching_distributed.py "
            "for the expected pattern."
        )

    objective_type = objective_args.objective_type
    objective_kwargs = objective_args.objective_kwargs or {}

    if objective_type == "miplib2017":
        return MIPLIB2017ObjectiveFunction(miplib_input_args=input_args, **objective_kwargs)
    if objective_type == "matching":
        return MatchingSolverDualObjectiveFunction(
            matching_input_args=input_args, gamma=solver_args.gamma, **objective_kwargs
        )
    raise ValueError(f"Objective type {objective_type} not supported")


def run_solver(
    input_args: BaseInputArgs,
    solver_args: SolverArgs,
    compute_args: ComputeArgs,
    objective_args: ObjectiveArgs,
) -> SolverResult:
    """Run the LP solver with the given configuration."""
    host_device = compute_args.host_device

    input_args = transfer_tensors_to_device(input_args, host_device)

    objective = build_objective(input_args, solver_args, compute_args, objective_args)

    solver = AcceleratedGradientDescent(
        max_iter=solver_args.max_iter,
        gamma=solver_args.gamma,
        initial_step_size=solver_args.initial_step_size,
        max_step_size=solver_args.max_step_size,
        gamma_decay_type=solver_args.gamma_decay_type,
        gamma_decay_params=solver_args.gamma_decay_params,
        save_primal=solver_args.save_primal,
    )

    initial_dual = (
        torch.load(solver_args.initial_dual_path)
        if solver_args.initial_dual_path is not None
        else torch.zeros_like(input_args.b_vec)
    )
    initial_dual = initial_dual.to(host_device)

    return solver.maximize(objective, initial_dual)
