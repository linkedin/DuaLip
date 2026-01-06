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
    \sum_{i \in T_1}\sum_k x_{ik} - \sum_{i \in T_2}\sum_k x_{ik} \le \delta \\
   -\sum_{i \in T_1}\sum_k x_{ik} + \sum_{i \in T_2}\sum_k x_{ik} \le \delta
  \end{array}

Together these enforce :math:`\big|\sum_{i \in T_1,k} x_{ik} - \sum_{i \in T_2,k} x_{ik}\big|
\le \delta`. Setting :math:`\delta=0` enforces exact parity; a small :math:`\delta>0` allows a
bounded imbalance.

Augmenting A and b
------------------
Using the notation in :doc:`matching`, the vector/matrix form of the optimization remains the same,
but the constraint matrix :math:`A` and vector :math:`b` are augmented as follows:

#. Keep the :math:`K` rows for movie limits :math:`\sum_i x_{ik} \le b_k`.
#. Add one row with coefficient :math:`+1` on variables :math:`x_{ik}` for :math:`i \in T_1`
   and :math:`-1` for :math:`i \in T_2`, with RHS :math:`\delta`.
#. Add one row with coefficient :math:`-1` on variables :math:`x_{ik}` for :math:`i \in T_1`
   and :math:`+1` for :math:`i \in T_2`, with RHS :math:`\delta`.

Equivalently, if :math:`b=(b_1,\ldots,b_K)` in :doc:`matching`, here we use
:math:`b'=(b_1,\ldots,b_K,\delta,\delta)`.

.. note::
  To include the fairness constraints described here, add the two fairness rows to :math:`A`
  and append :math:`(\delta, \delta)` to :math:`b` before saving the tensors. The rest of the
  pipeline (data preparation, solver invocation, and reading results) is unchanged from
  :doc:`matching`.

