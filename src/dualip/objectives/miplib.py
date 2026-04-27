from dataclasses import dataclass
from typing import Optional

import torch

from dualip.objectives.base import BaseInputArgs, BaseObjective
from dualip.projections.base import ProjectionEntry, project
from dualip.types import ObjectiveResult


@dataclass
class MIPLIBInputArgs(BaseInputArgs):
    """
    Input arguments specific to MIPLIB2017 objective function.
    """

    A: torch.Tensor
    c: torch.Tensor
    projection_map: dict[str, ProjectionEntry]
    b_vec: torch.Tensor
    equality_mask: Optional[torch.Tensor]

    def __post_init__(self):
        super().__post_init__()
        pass


class MIPLIB2017ObjectiveFunction(BaseObjective):
    """
    Computes dual gradient, objective, and regularization penalty
    for a (single-GPU) matching problem.
    """

    def __init__(
        self,
        miplib_input_args: MIPLIBInputArgs,
        gamma: float = 1.0,
        use_jacobi_precondition: bool = False,
    ):
        self.gamma = gamma
        self.A = miplib_input_args.A
        # Store CSR and CSC versions of A for efficient computations if needed
        self.A_csr = self.A.to_sparse_csr() if self.A.is_sparse else self.A
        self.A_csc_T = self.A.to_sparse_csc().transpose(0, 1) if self.A.is_sparse else self.A.T
        self.c = miplib_input_args.c
        self.b_vec = miplib_input_args.b_vec
        self.projection_map = miplib_input_args.projection_map
        self.equality_mask = miplib_input_args.equality_mask
        self.lower, self.upper = self._construct_variable_lower_upper_bound()
        self.use_jacobi_precondition = use_jacobi_precondition

        if self.use_jacobi_precondition:
            if self.A.is_sparse:
                raise NotImplementedError("Jacobi preconditioning is not implemented for sparse matrices")
            else:
                row_norms = torch.norm(self.A, dim=1, keepdim=True)
            # Avoid division by zero for rows with all zeros
            self.row_norms = torch.where(row_norms == 0, torch.ones_like(row_norms), row_norms).squeeze()
        else:
            self.row_norms = None

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma

    def calculate(self, dual_val: torch.Tensor, save_primal: bool = False, **kwargs) -> ObjectiveResult:
        """
        Compute dual gradient, objective, and reg penalty.

        Args:
            dual_val: current dual variables
            save_primal: if True, save the primal variable

        Returns:
            ObjectiveResult
        """

        if self.row_norms is not None:
            dual_val = 1 / self.row_norms * dual_val

        z = -1.0 / self.gamma * (self.A.T @ dual_val + self.c)

        # Apply projection on z based on projection_map
        projected_sol = z.clone()
        for _, proj_item in self.projection_map.items():
            indices = torch.tensor(proj_item.indices, dtype=torch.long, device=z.device)
            proj_type = proj_item.proj_type
            proj_params = proj_item.proj_params

            # Create projection operator
            proj_fn = project(proj_type, **proj_params)

            # Apply projection to the specified indices
            if indices.numel() > 0:
                projected_sol[indices] = proj_fn(projected_sol[indices])

        if self.row_norms is not None:
            dual_gradient = 1 / self.row_norms * (self.A_csr @ projected_sol - self.b_vec)
        else:
            dual_gradient = self.A_csr @ projected_sol - self.b_vec

        reg_penalty = self.gamma / 2.0 * torch.norm(projected_sol) ** 2

        dual_obj = self.c @ projected_sol + reg_penalty + dual_val @ (self.A_csr @ projected_sol - self.b_vec)
        primal_obj = self.c @ projected_sol
        result = ObjectiveResult(
            dual_gradient=dual_gradient,
            dual_objective=dual_obj,
            reg_penalty=reg_penalty,
        )
        if save_primal:
            result.primal_var = projected_sol
            result.primal_objective = primal_obj
        return result

    def _construct_variable_lower_upper_bound(self):
        lower = torch.full_like(self.c, float("nan"))
        upper = torch.full_like(self.c, float("nan"))

        for _, proj_item in self.projection_map.items():
            indices = torch.tensor(proj_item.indices, dtype=torch.long, device=self.c.device)
            if "l" in proj_item.proj_params:
                lower[indices] = proj_item.proj_params["l"]
            if "u" in proj_item.proj_params:
                upper[indices] = proj_item.proj_params["u"]
        return lower, upper

    def _clamp_x_bound_duals(self, x_bound_duals, l_mask_exists, u_mask_exists):
        r"""
        This function projects the dual variables corresponding to the primal
        variables bounds (l_i, u_i).
        The projection is done according to the following rules:
            1. If only the lower bound exists (l_i is not NaN, u_i is NaN),
               clamp the dual variable to be >= 0.
            2. If only the upper bound exists (l_i is NaN, u_i is not NaN),
               clamp the dual variable to be <= 0.
            3. If neither bound exists (both l_i and u_i are NaN),
               set the dual variable to 0.
            4. If both bounds exist (neither l_i nor u_i are NaN),
               leave the dual variable unchanged.
        See the definition of set \Lambda defined in the PLDP paper under
        Equation (1).
        """
        result = x_bound_duals.clone()

        # Case 1: l=True, u=False => clamp with min=0
        mask_min = l_mask_exists & (~u_mask_exists)
        result[mask_min] = torch.clamp(result[mask_min], min=0)

        # Case 2: l=False, u=True => clamp with max=0
        mask_max = (~l_mask_exists) & u_mask_exists
        result[mask_max] = torch.clamp(result[mask_max], max=0)

        # Case 3: l=False, u=False => set to 0
        mask_zero = (~l_mask_exists) & (~u_mask_exists)
        result[mask_zero] = 0

        # Case 4: l=True, u=True => unchanged
        return result

    def calculate_convergence_bound(
        self, dual_val: torch.Tensor, x: torch.Tensor = None, optimal_primal_obj=None, tol: float = 1e-4
    ) -> bool:
        """
        PDLP convergence test (see Eq. 6a–6b in Applegate, 2022).
        This computes the dual and primal objective values WITHOUT regularization.
        It does so by plugging in the current dual and primal values into the problem and
        computing the duality gap and feasibility conditions.

        Parameters
        ----------
        dual_val : torch.Tensor, shape (m,)
            Current dual variable.
        x : torch.Tensor, optional, shape (n,)
            A primal solution.
        optimal_primal_obj: float, optional.
            Optimal primal objective value, if known.
        tol : float, default=1e-4
            Convergence tolerance.
        Returns
        -------
        bool
            True if PDLP stopping criterion is satisfied (gap and feasibility
            are within tolerance), False otherwise.
        """
        if self.row_norms is not None:
            dual_val = 1 / self.row_norms * dual_val

        # Compute reduced cost: r = c + A^T λ
        r = self.c + self.A.t().mv(dual_val)

        if x is None:
            # construct based on reduced cost (in the future we can call different primal recovery methods)
            x = torch.where(r >= 0, self.lower, self.upper)
            if torch.isnan(x).any():
                raise ValueError("Unbounded x.")

        # Decompose dual residual into positive and negative components
        lambda_neg = torch.clamp(r, max=0.0)
        lambda_pos = torch.clamp(r, min=0.0)

        # Compute dual objective: d = -bλ + <λ-, u> + <λ+, l>
        u_exists_mask = ~torch.isnan(self.upper)
        l_exists_mask = ~torch.isnan(self.lower)

        lambda_u = torch.dot(lambda_neg[u_exists_mask], self.upper[u_exists_mask])
        lambda_l = torch.dot(lambda_pos[l_exists_mask], self.lower[l_exists_mask])
        d = -torch.dot(self.b_vec, dual_val) + lambda_u + lambda_l

        # Compute primal objective: p = cx
        p = torch.dot(self.c, x)

        # Compute duality gap: |p - d| / (1 + |p| + |d|)
        gap_upperbound = torch.abs(p - d) / (1.0 + torch.abs(p) + torch.abs(d))

        if optimal_primal_obj is not None:
            gap_lower_bound = torch.abs(p - optimal_primal_obj) / (1.0 + torch.abs(p) + torch.abs(optimal_primal_obj))
        else:
            gap_lower_bound = torch.tensor(float("nan"))

        # Compute primal feasibility:
        Ax_minus_b = self.A.mv(x) - self.b_vec

        if self.equality_mask is None:
            row_violation = torch.relu(Ax_minus_b)
        else:
            row_violation = torch.where(self.equality_mask, Ax_minus_b.abs(), torch.relu(Ax_minus_b))

        primal_feas = torch.linalg.vector_norm(row_violation) / (1.0 + torch.linalg.vector_norm(self.b_vec))

        x_bound_duals = self._clamp_x_bound_duals(-r, l_exists_mask, u_exists_mask)
        dual_feas = torch.linalg.vector_norm(r + x_bound_duals) / (1.0 + torch.linalg.vector_norm(self.c))

        converged = (gap_upperbound <= tol) and (primal_feas <= tol) and (dual_feas <= tol)
        return gap_upperbound, gap_lower_bound, primal_feas, dual_feas, converged
