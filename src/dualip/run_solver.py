from dataclasses import fields
from typing import Optional

import torch

from dualip.objectives.base import BaseInputArgs
from dualip.objectives.matching import (
    MatchingSolverDualObjectiveFunction,
    MatchingSolverDualObjectiveFunctionDistributed,
)
from dualip.objectives.miplib import MIPLIB2017ObjectiveFunction
from dualip.optimizers.agd import AcceleratedGradientDescent, SolverResult
from dualip.types import ComputeArgs, ObjectiveArgs, SolverArgs
from dualip.utils.mlflow_utils import MLflowConfig, log_hyperparameters, mlflow_run_context


def transfer_tensors_to_device(input_args: BaseInputArgs, device: str):
    """
    Transfer all tensor fields in input_args to the specified device.

    Args:
        input_args: The input arguments object
        device: The target device (e.g., 'cuda:0', 'cpu')

    Returns:
        A new instance of input_args with all tensors transferred to device
    """
    # Get all field names from the dataclass
    field_names = [field.name for field in fields(input_args)]

    # Create a dictionary of field values with tensors transferred to device
    field_values = {}
    for field_name in field_names:
        value = getattr(input_args, field_name)
        if isinstance(value, torch.Tensor):
            field_values[field_name] = value.to(device)
        else:
            field_values[field_name] = value

    # Create a new instance of the same class with transferred tensors
    return type(input_args)(**field_values)


def build_objective(
    input_args: BaseInputArgs, solver_args: SolverArgs, compute_args: ComputeArgs, objective_args: ObjectiveArgs
):
    objective = None
    host_device = compute_args.host_device
    compute_device_num = compute_args.compute_device_num

    objective_type = objective_args.objective_type
    objective_kwargs = objective_args.objective_kwargs

    if objective_type == "miplib2017":
        objective_kwargs = objective_kwargs or {}
        objective = MIPLIB2017ObjectiveFunction(miplib_input_args=input_args, **objective_kwargs)
    elif objective_type == "matching":
        if compute_device_num == 1:
            objective = MatchingSolverDualObjectiveFunction(matching_input_args=input_args, gamma=solver_args.gamma)
        else:
            compute_devices = [f"cuda:{i}" for i in range(compute_args.compute_device_num)]
            objective = MatchingSolverDualObjectiveFunctionDistributed(
                matching_input_args=input_args,
                gamma=solver_args.gamma,
                host_device=host_device,
                compute_devices=compute_devices,
            )

    else:
        raise ValueError(f"Objective type {objective_type} not supported")
    return objective


def run_solver(
    input_args: BaseInputArgs,
    solver_args: SolverArgs,
    compute_args: ComputeArgs,
    objective_args: ObjectiveArgs,
    mlflow_config: Optional[MLflowConfig] = None,
) -> SolverResult:
    """
    Run the LP solver with the given configuration.

    Args:
        input_args: Input data arguments
        solver_args: Solver configuration arguments
        compute_args: Compute configuration arguments
        objective_args: Objective function configuration arguments
        mlflow_config: Optional MLflow configuration for logging

    Returns:
        The solver result.
    """
    # Set up MLflow if configured
    if mlflow_config is None:
        mlflow_config = MLflowConfig(enabled=False)

    with mlflow_run_context(mlflow_config):
        # Log hyperparameters if MLflow is enabled
        if mlflow_config.enabled and mlflow_config.log_hyperparameters:
            hyperparams = {
                "solver": solver_args.__dict__,
                "objective": objective_args.__dict__,
            }
            log_hyperparameters(hyperparams)

        host_device = compute_args.host_device

        # Transfer all tensor fields in input_args to host device
        input_args = transfer_tensors_to_device(input_args, host_device)

        # Initialize objective
        objective = build_objective(input_args, solver_args, compute_args, objective_args)

        # Create solver
        solver = AcceleratedGradientDescent(
            initial_step_size=solver_args.initial_step_size,
            max_iter=solver_args.max_iter,
            max_step_size=solver_args.max_step_size,
            gamma=solver_args.gamma,
            gamma_decay_type=solver_args.gamma_decay_type,
            gamma_decay_params=solver_args.gamma_decay_params,
            save_primal=solver_args.save_primal,
        )

        # Initialize dual
        initial_dual = (
            torch.load(solver_args.initial_dual_path)
            if solver_args.initial_dual_path is not None
            else torch.zeros_like(input_args.b_vec).requires_grad_(False)
        )
        initial_dual = initial_dual.to(host_device)

        solver_result = solver.maximize(objective, initial_dual)

        use_jacobi_precondition = getattr(objective, "use_jacobi_precondition", None)
        if use_jacobi_precondition:
            dual_val = solver_result.dual_val
            dual_grad = solver_result.objective_result.dual_gradient

            inverted_dual_val, inverted_dual_grad = objective.invert_jacobi_precondition(dual_val, dual_grad)

            solver_result.dual_val = inverted_dual_val
            solver_result.objective_result.dual_gradient = inverted_dual_grad

        return solver_result
