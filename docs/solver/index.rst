.. _solver:

The DuaLip Solver
=================

Problem Statement
-----------------

In a typical recommender system problem, we denote users by :math:`i = 1, \ldots ,I` and items by :math:`k = 1, \ldots, K`. Let 
:math:`x_{ik}` denote any association between user :math:`i` and item :math:`k`, and be the variable of interest. For example, 
:math:`x_{ik}` can be the probability of displaying item :math:`k` to user :math:`i`. The vectorized version is denoted by 
:math:`x = (x_1, ..., x_I)` where :math:`x_i = (x_{i1}, ..., x_{iK})`. 

DuaLip solves Linear Programs (LPs) of the following form:

.. math::
  \begin{array}{ll}
    \mbox{minimize} & c^T x \\
    \mbox{subject to} & A x \leq b \\
    & x_i \in \mathcal{C}_i \;\; \text{for all}\; = 1,\ldots, I
  \end{array}

where :math:`A_{m \times n}` is the constraint matrix, :math:`b_{m \times 1}` is the constraint vector and :math:`\mathcal{C}_i` are uniformly
compact polytopes. :math:`x \in \mathbb{R}^n` is the vector of optimization variables, where :math:`n = IK`. 

.. _probsolution:

Problem Solution
----------------

We briefly outline the solution mechanism here. For more details, please see `Basu et al. (2020)
<http://proceedings.mlr.press/v119/basu20a/basu20a.pdf>`_.
To solve the problem, we introduce the perturbed problem:

.. math::
  \begin{array}{ll}
    \mbox{minimize} & x^T c  + \frac{\gamma}{2}x^T x \\
    \mbox{subject to} & A x \leq b \\
    & x_i \in \mathcal{C}_i \;\; \text{for all}\; = 1,\ldots, I
  \end{array}

where :math:`\gamma > 0` controls the tradeoff between problem approximation and the smoothness of the dual objective function.
To make the above problem amenable to first order methods, we consider the Lagrangian dual:

.. math::
    g_{\gamma}(\lambda) = \min_{x \in \mathcal C} ~~ \left\{ c^T x + \frac{\gamma}{2} x^T x + \lambda^T(Ax-b) \right\},

where :math:`\mathcal{C} = \Pi_{i=1}^I \mathcal{C}_i`. Now, by strong duality, the optimum objective :math:`g_{\gamma}^*` of the dual

.. math::
    g_{\gamma}^*:=\max_{\lambda \geq 0} ~ g_{\gamma}(\lambda)

is the minimum of the above problem. We can show that :math:`\lambda \mapsto g_{\gamma}(\lambda)` is differentiable and the
gradient is Lipschitz continuous. Moreover, by Danskin's Theorem the gradient can be explicitly expressed as,

.. math::
    \nabla g_{\gamma}(\lambda) = A x_{\gamma}^*(\lambda) -b

where,

.. math::
    x_{\gamma}^*(\lambda) &= \text{argmin}_{x \in \mathcal C} ~~ \left\{ c^T x + \frac{\gamma}{2} x^T x + \lambda^T(Ax-b) \right\}  \\
    & = \big\{
    \Pi_{\mathcal{C}_i}[-\frac{1}{\gamma}({A_i}^T\lambda + c_i)]
    \big\}_{i=1}^I

where :math:`\Pi_{\mathcal{C}_i}(\cdot)` is the Euclidean projection operator onto  :math:`\mathcal{C}_i`, and, :math:`A_i`, :math:`c_i` are the
parts of :math:`A` and :math:`c` corresponding to :math:`x_i`. Based on this we use a first-order gradient method as the main optimizer to
solve the problem. It can also be shown that the solution obeys certain bounds to the true solution :math:`g_0(\lambda)` and 
in fact the exact solution of the LP can be obtained if :math:`\gamma` is small enough. 
For more details, please refer to `Basu et al. (2020)
<http://proceedings.mlr.press/v119/basu20a/basu20a.pdf>`_.


.. _algorithm:

The Algorithm
-------------

The overall algorithm can now be written as:

1. Start with an initial :math:`\lambda`.
2. Get Primal: :math:`x_{\gamma}^*(\lambda)`.
3. Get Gradient: :math:`Ax_{\gamma}^*(\lambda) - b`.
4. Update :math:`\lambda` via appropriate mechanisms.
5. Continue till converge.
   
We currently support `Accelerated Gradient Ascent <https://www.ceremade.dauphine.fr/~carlier/FISTA>`_ as the maximizer though the solver is easily extensible to other optimization algorithms.

.. _constraints:

Constraint Sets :math:`\mathcal{C}_i`
-------------------------------------
In this current version of the solver we support a wide variety of constraints types :math:`\mathcal{C}_i`, 
such as:

1. Unit Box: :math:`\mathcal{C}_i = \big\{ x \in \mathbb{R}^K : 0 \leq x_k \leq 1\big\}`
2. Simplex-E: :math:`\mathcal{C}_i = \big\{ x \in \mathbb{R}^K : x_1 + ... + x_K = 1, \;\; x_k \geq 0\big\}`
3. Simplex-I: :math:`\mathcal{C}_i = \big\{ x \in \mathbb{R}^K : x_1 + ... + x_K \leq 1, \;\; x_k \geq 0\big\}`
4. r-Simplex-E: :math:`\mathcal{C}_i = \big\{ x \in \mathbb{R}^K : x_1 + ... + x_K = r, \;\; x_k \geq 0\big\}`
5. r-Simplex-I: :math:`\mathcal{C}_i = \big\{ x \in \mathbb{R}^K : x_1 + ... + x_K \leq r, \;\; x_k \geq 0\big\}`

Here :math:`E` and :math:`I` stands for equality and inequality.


To execute step 2 of the overall algorithm, we need a projection operation on these constraint sets.
In our solver, we have implemented highly efficient projection algorithms to make step 2 extremely fast. The different sets have 
different customized algorithms to make the overall system highly efficient. For more details on 
these projection algorithms please see Section 3 of `Ramanath et al. (2021)
<https://arxiv.org/abs/2103.05277>`_.


.. .. _adaptive_smoothing:

.. Adaptive Smoothing
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. The smoothness of :math:`g_\gamma` decreases as the number of constraints increases. 
.. A small :math:`\gamma` makes the optimizer's convergence prohibitively slow, while a large :math:`\gamma` reduces the accuracy of 
.. the solution. The solver allows for a basic adaptive smoothing where the :math:`\gamma` is reduced by a under-defined factor at specified intervals.


Data Sharding (Multi-GPU)
-------------------------------------

For large matching problems, DuaLip can distribute computation across multiple GPUs by sharding the input data along the column dimension of the constraint matrix. When ``compute_device_num > 1``, the solver builds a distributed objective that wraps a single‑GPU objective on each device and coordinates reductions on a host device.

- Sharding of inputs: Matrices ``A`` and ``c`` are partitioned into roughly equal contiguous blocks across the available compute devices (e.g., ``cuda:0``, ``cuda:1``, ...). This balances work by splitting the number of columns as evenly as possible. Each shard is then moved to its target device. The per‑device projection map is derived from the global projection map by remapping global column indices to local indices for that shard. Only projections that touch columns present on the device are kept.
- Per‑iteration execution: The current dual vector :math:`\lambda` is transferred to each compute device. Each device computes its local dual gradient contribution, local dual objective component, and regularization penalty using the single‑GPU matching objective on its shard. Partial results are first accumulated on the host device, then synchronized and summed across processes using NCCL all‑reduce. The final distributed gradient subtracts :math:`b` and is used by the optimizer exactly as in the single‑GPU case.

This design keeps projection logic local to each shard, minimizes inter‑GPU communication to a small number of vector/tensor reductions per iteration, and scales naturally with the number of GPUs. 

Implementation Note
-------------------
As we will discuss in the :ref:`Supported LPs <supported_lps>` section, currently the distributed objective is implemented for matching problems where the constraint matrix is a block-diagonal matrix. In this case, the inputs must be CSC‑format sparse tensors. Sharding operates on columns to align with how projections are applied per column group. For custom objective functions, the user needs to implement the parallelism themselves.
