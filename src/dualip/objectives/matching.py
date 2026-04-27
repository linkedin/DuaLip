from dataclasses import dataclass
from operator import add, mul

import torch
import torch.distributed as dist

from dualip.objectives.base import BaseInputArgs, BaseObjective, ObjectiveResult
from dualip.projections.base import ProjectionEntry, project
from dualip.utils.objective_utils import calc_grad
from dualip.utils.sparse_utils import apply_F_to_columns, elementwise_csc, left_multiply_sparse, row_sums_csc


@dataclass
class MatchingInputArgs(BaseInputArgs):
    """
    Input arguments specific to Matching objective function.
    """

    A: torch.Tensor
    c: torch.Tensor
    projection_map: dict[str, ProjectionEntry]
    b_vec: torch.Tensor
    equality_mask: torch.Tensor = None


class MatchingSolverDualObjectiveFunction(BaseObjective):
    """
    Computes dual gradient, objective, and regularization penalty
    for a (single-GPU) matching problem.
    """

    def __init__(
        self,
        matching_input_args: MatchingInputArgs,
        gamma: float,
        batching: bool = True,
    ):
        self.A = matching_input_args.A
        self.c = matching_input_args.c
        self.gamma = gamma
        self.b_vec = matching_input_args.b_vec
        self.projection_map = matching_input_args.projection_map
        # If b_vec is provided, then this is a total single device objecitve function, otherwise this class
        # is being used to encapsulate single-GPU computation in the distributed setting.
        self.is_distributed = self.b_vec is None
        self.equality_mask = matching_input_args.equality_mask

        device = self.A.device

        # Batching variables
        self._thresholds = []
        self._bucket_ids = None

        # Precompute c_rescaled = -c / gamma
        self.c_rescaled = -1.0 / gamma * self.c

        # Build buckets
        self.buckets = {}
        for proj_key, proj_item in self.projection_map.items():
            indices = torch.tensor(proj_item.indices, dtype=torch.int32, device=device)
            proj_type = proj_item.proj_type
            proj_params = proj_item.proj_params
            if batching:
                self.buckets[proj_key] = (self._compute_buckets(indices), proj_type, proj_params)
            else:
                self.buckets[proj_key] = ([indices], proj_type, proj_params)

        # Pre-allocate a CSC tensor to hold intermediate results
        self.intermediate = torch.sparse_csc_tensor(
            self.A.ccol_indices(),
            self.A.row_indices(),
            torch.zeros_like(self.A.values()),
            size=self.A.size(),
        )

    def _compute_buckets(self, indices: list[torch.Tensor]) -> list[list[torch.Tensor]]:
        device = self.A.device
        if not self._thresholds:
            ccol_ptr = self.A.ccol_indices()

            # build thresholds
            self._thresholds = [0]
            i = 1
            max_nnz = self.A.size(0)
            while 2**i <= max_nnz:
                self._thresholds.append(2**i)
                i += 1
            self._thresholds.append(max_nnz + 1)

            # per-column nnz counts
            lengths = ccol_ptr[1:] - ccol_ptr[:-1]
            th_tensor = torch.tensor(self._thresholds, dtype=lengths.dtype, device=device)
            self._bucket_ids = torch.bucketize(lengths.to(device), th_tensor)

        indices = torch.as_tensor(indices, dtype=torch.int32, device=device)
        proj_bucket_ids = self._bucket_ids[indices]
        # for each nnz bucket, gather columns of this projection type that fall into that bucket
        buckets = []
        for j in range(1, len(self._thresholds)):
            bucket = indices[proj_bucket_ids == j]
            if bucket.numel() > 0:
                buckets.append(bucket)
        return buckets

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma
        self.c_rescaled = -1.0 / gamma * self.c

    def calculate(self, dual_val: torch.Tensor, save_primal: bool = False, **kwargs) -> ObjectiveResult:
        """
        Compute dual gradient, objective, and reg penalty.

        Args:
            dual_val: current dual variables
            save_primal: if True, save the primal variable

        Returns:
            ObjectiveResult
        """
        # -dual_val/gamma
        scaled = -1.0 / self.gamma * dual_val

        # intermediate = A * scaled
        left_multiply_sparse(scaled, self.A, output_tensor=self.intermediate)

        # intermediate += c_rescaled
        elementwise_csc(self.intermediate, self.c_rescaled, add, output_tensor=self.intermediate)

        # apply each projection
        for _, proj_item in self.buckets.items():
            buckets = proj_item[0]
            proj_type = proj_item[1]
            proj_params = proj_item[2]
            fn = project(proj_type, **proj_params)
            apply_F_to_columns(self.intermediate, fn, buckets, output_tensor=self.intermediate)

        # dual gradient = row sums of A * intermediate
        grad = row_sums_csc(elementwise_csc(self.A, self.intermediate, mul))

        # reg penalty = (gamma/2) * ||intermediate.values||^2
        vals = self.intermediate.values()
        reg_penalty = (self.gamma / 2) * torch.norm(vals) ** 2

        # dual objective = c * intermediate.values
        dual_obj = torch.dot(self.c.values(), vals)
        primal_obj = dual_obj.clone()
        primal_var = vals

        if not self.is_distributed and self.b_vec is not None:
            grad, dual_obj = calc_grad(grad, dual_obj, dual_val, self.b_vec, reg_penalty)

            dual_val_times_grad = torch.dot(dual_val, grad)
            max_pos_slack = max(torch.max(grad), 0)
            sum_pos_slack = torch.relu(grad).sum()

            obj_result = ObjectiveResult(
                dual_gradient=grad,
                dual_objective=dual_obj,
                reg_penalty=reg_penalty,
                dual_val_times_grad=dual_val_times_grad,
                max_pos_slack=max_pos_slack,
                sum_pos_slack=sum_pos_slack,
            )
        else:
            obj_result = ObjectiveResult(
                dual_gradient=grad,
                dual_objective=dual_obj,
                reg_penalty=reg_penalty,
            )
        if save_primal:
            obj_result.primal_var = primal_var
            obj_result.primal_objective = primal_obj
        return obj_result


class MatchingSolverDualObjectiveFunctionDistributed(BaseObjective):
    """
    Distributed wrapper for matching objective using PyTorch's multi-process model.

    Design:
        - Each rank (process) creates this with its LOCAL data partition
        - Each rank computes gradients on its own GPU (cuda:rank)
        - Results are aggregated via NCCL reduce to rank 0 (cuda:0)
        - Only rank 0 returns meaningful results; other ranks return dummy values

    Usage:
        1. Split data before initializing torch.distributed
        2. Each rank takes its partition by index
        3. Each rank creates this objective with its local data
        4. Solver coordinates distributed computation via rank parameter

    Example:
        >>> # Each rank does this:
        >>> rank = torch.distributed.get_rank()
        >>> A_local = A_splits[rank].to(f"cuda:{rank}")
        >>> c_local = c_splits[rank].to(f"cuda:{rank}")
        >>> local_args = MatchingInputArgs(A_local, c_local, pm_local, b_vec=None)
        >>> objective = MatchingSolverDualObjectiveFunctionDistributed(
        ...     local_args, b_vec, gamma, host_device="cuda:0"
        ... )
    """

    def __init__(
        self,
        local_matching_input_args: MatchingInputArgs,
        b_vec: torch.Tensor,
        gamma: float,
        host_device: torch.device,
        batching: bool = True,
    ):
        """
        Initialize distributed objective with local data partition.

        Args:
            local_matching_input_args: Local data partition for this rank.
                Must have b_vec=None since b_vec is shared across all ranks.
            b_vec: Global constraint vector (same across all ranks).
                Will be moved to host_device for final aggregation.
            gamma: Regularization parameter.
            host_device: Device for aggregation, typically "cuda:0".
                All ranks send results here via NCCL reduce.
            batching: Whether to use batched projection operations. Default True.
        """
        self.gamma = gamma
        self.host_device = host_device
        self.b_vec = b_vec.to(host_device)
        self.equality_mask = local_matching_input_args.equality_mask

        # Create single-GPU objective with local data
        self.local_objective = MatchingSolverDualObjectiveFunction(local_matching_input_args, gamma, batching)

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma
        self.local_objective.set_gamma(gamma)

    def calculate(
        self,
        dual_val: torch.Tensor,
        save_primal: bool = False,
        rank: int = 0,
        **kwargs,
    ) -> ObjectiveResult:
        """Compute and reduce gradients/objectives across all GPUs."""
        if save_primal:
            raise NotImplementedError("save_primal=True is not yet supported in distributed mode")

        # dual_val is on cuda:rank (each rank has it on its own device)
        # local_objective data is also on cuda:rank
        # Compute local partition
        objective_result = self.local_objective.calculate(dual_val, save_primal=False)

        # Keep results on local device (cuda:rank) for NCCL reduce
        # NCCL expects each rank to have tensor on its own GPU
        grad_local = objective_result.dual_gradient
        obj_local = objective_result.dual_objective
        reg_local = objective_result.reg_penalty

        # ALL ranks participate in reduce operations
        # Each tensor is on cuda:rank, NCCL handles cross-GPU communication
        # After reduce, only rank 0 has the aggregated result (on cuda:0)
        dist.reduce(grad_local, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(obj_local, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(reg_local, dst=0, op=dist.ReduceOp.SUM)

        # ALL ranks synchronize
        dist.barrier()

        # Only rank 0 has the aggregated results and computes final objective
        if rank == 0:
            # After reduce, grad_local, obj_local, reg_local are on cuda:0 with aggregated values
            # dual_val is already on cuda:0 (rank 0's device)
            grad = grad_local - self.b_vec
            dual_val_times_grad = torch.dot(dual_val, grad)
            obj = obj_local + reg_local + dual_val_times_grad

            max_pos_slack = max(torch.max(grad), 0)
            sum_pos_slack = torch.relu(grad).sum()

            obj_result = ObjectiveResult(
                dual_gradient=grad,
                dual_objective=obj,
                reg_penalty=reg_local,
                dual_val_times_grad=dual_val_times_grad,
                max_pos_slack=max_pos_slack,
                sum_pos_slack=sum_pos_slack,
            )

            return obj_result
        else:
            # Non-zero ranks return a dummy result (won't be used by optimizer)
            # Results are still on cuda:rank (not cuda:0)
            return ObjectiveResult(
                dual_gradient=torch.zeros_like(grad_local),
                dual_objective=torch.tensor(0.0, device=grad_local.device),
                reg_penalty=torch.tensor(0.0, device=grad_local.device),
            )
