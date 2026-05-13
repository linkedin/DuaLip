.. _matching_complex:
A Matching Problem with Fairness Constraints
============================================
Buidling on the problem formulation descrived in :doc:`matching`, we now wish to consider a matching probelem with fairness constraints. We assume the base matching formulation, data layout,
and solver workflow described in :doc:`matching`, and focuses only on adding fairness constraints between two
user types and demonstrate how the objective function class can be easily extended to support more complex constraints.

Fairness across user types
--------------------------
Assume we have two user types: users with IDs :math:`1,2` belong to type 1 and users with IDs
:math:`3,4` belong to type 2. We wish to ensure type 1 and type 2 receive almost the same total
number of recommendations (in expectation). Let

.. math::
  T_1 = \{1,2\}, \qquad T_2 = \{3,4\}.

Define a tolerance :math:`\delta \ge 0` that bounds the allowed deviation between the total
recommendations to the two types. We impose the following linear fairness constraints:

.. math::
  \begin{array}{ll}
    \frac{1}{|T_1|}\sum_{i \in T_1}\sum_k x_{ik} - \frac{1}{|T_2|}\sum_{i \in T_2}\sum_k x_{ik} \le \delta \\
   -\frac{1}{|T_1|}\sum_{i \in T_1}\sum_k x_{ik} + \frac{1}{|T_2|}\sum_{i \in T_2}\sum_k x_{ik} \le \delta
  \end{array}

Together these enforce :math:`\big|\frac{1}{|T_1|}\sum_{i \in T_1,k} x_{ik} - \frac{1}{|T_2|}\sum_{i \in T_2,k} x_{ik}\big|
\le \delta`. Setting :math:`\delta=0` enforces exact parity; a small :math:`\delta>0` allows a
bounded imbalance.

Augmenting A and b
------------------
Using the notation in :doc:`matching`, the vector/matrix form of the optimization remains the same,
but the constraint matrix :math:`A` and vector :math:`b` are augmented as follows:

#. Keep the :math:`K` rows for movie limits :math:`\sum_i x_{ik} \le b_k`.

#. Add one row with coefficient :math:`\frac{1}{|T_1|}` on variables :math:`x_{ik}` for :math:`i \in T_1`
   and :math:`-\frac{1}{|T_2|}` for :math:`i \in T_2`, with RHS :math:`\delta`.

#. Add one row with coefficient :math:`-\frac{1}{|T_1|}` on variables :math:`x_{ik}` for :math:`i \in T_1`
   and :math:`\frac{1}{|T_2|}` for :math:`i \in T_2`, with RHS :math:`\delta`.

For steps 2 and 3, we add the following function to the matching objective function class:

.. code-block:: python

    def _build_fairness_constraints(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_cols = self.A.size(1)
        group_1_size = int(num_cols * self.group_ratio)
        group_1_size = max(0, min(group_1_size, num_cols))
        group_2_size = num_cols - group_1_size

        # Split columns into first group and second group (remaining)
        A_blocks = split_csc_by_cols(self.A, [group_1_size, group_2_size]) if num_cols > 0 else [self.A, self.A]
        A_group_1, A_group_2 = A_blocks

        # Normalize the first group 
        A_group_1 = 1/group_1_size * A_group_1
        # Negate and normalizethe second group's columns
        A_group_2 = -1/group_2_size * A_group_2
        # Concatenate back column-wise to match original shape/pattern
        A_fairness = hstack_csc([A_group_1, A_group_2])

        return A_fairness

where A_fairness is passed to the objective function as an additional constraint matrix.

Equivalently, if :math:`b=(b_1,\ldots,b_K)` in :doc:`matching`, here we use
:math:`b'=(b_1,\ldots,b_K,\delta,\delta)`.

.. note::
  To include the fairness constraints described here, add the two fairness rows to :math:`A`
  and append :math:`(\delta, \delta)` to :math:`b` before saving the tensors. Since fairness constraints introduce 
  constraint structure not handled by the default matching objective function, we need to extend the objective function to support fairness constraints.


Extending the Gradient Calculation
----------------------------------
To extend the objective function to support fairness constraints, we use the :class:`MatchingSolverDualObjectiveFunction` class and override the :meth:`calculate` method.
The new gradient calculation is given by:

.. code-block:: python
   :emphasize-lines: 29-30,32-33,47-48,49-50

    def calculate(
        self,
        dual_val: torch.Tensor,
        gamma: float = None,
        save_primal: bool = False,
    ) -> ObjectiveResult:
        """
        Compute dual gradient, objective, and reg penalty.

        Args:
            dual_val: current dual variables
            gamma: regularization parameter
            save_primal: if True, save the primal variable

        Returns:
            ObjectiveResult
        """
        grad = torch.zeros_like(dual_val)

        if gamma is not None:
            self.gamma = gamma

        # -dual_val/gamma
        scaled = -1.0 / self.gamma * dual_val

        # intermediate = A * scaled
        left_multiply_sparse(scaled[:-2], self.A, output_tensor=self.intermediate)

        # intermediate += A_fairness * scaled
        elementwise_csc(self.intermediate, scaled[-2] * self.A_fairness, add, output_tensor=self.intermediate)

        # intermediate += -A_fairness * scaled
        elementwise_csc(self.intermediate, -1 * scaled[-1] * self.A_fairness, add, output_tensor=self.intermediate)

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
        grad[:-2] = row_sums_csc(elementwise_csc(self.A, self.intermediate, mul))
        grad[-2] = elementwise_csc(self.A_fairness, self.intermediate, mul).values().sum()
        grad[-1] = elementwise_csc(-1 * self.A_fairness, self.intermediate, mul).values().sum()

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

The above changes should be the main changes to the objective function to support fairness constraints.
Additional changes may be needed to the input arguments, and data preparation to support fairness constraints e.g., additional parameters such as group ration or tolerance. 
