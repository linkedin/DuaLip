.. _solver :

The Dualip Solver
=================

Problem Statement
-----------------

In a typical recommender system problem, we denote users by :math:`i = 1, \ldots ,I` and items by :math:`k = 1, \ldots, K`. Let 
:math:`x_{ik}` denote any association between user :math:`i` and item :math:`k`, and be the variable of interest. For example, 
:math:`x_{ik}` can be the probability of displaying item :math:`k` to user :math:`i`. The vectorized version is denoted by 
:math:`x = (x_1, ..., x_I)` where :math:`x_i = (x_{i1}, ..., x_{iK})`. 

Dualip solves Linear Programs (LPs) of the following form:

.. math::
  \begin{array}{ll}
    \mbox{minimize} & c^T x \\
    \mbox{subject to} & A x \leq b \\
    & x_i \in \mathcal{C}_i \;\; \text{for all}\; = 1,\ldots, I
  \end{array}

where :math:`A_{m \times n}` is the constraint matrix, :math:`b_{m \times 1}` is the constraint vector and the :math:`\mathcal{C}_i` are uniformly
compact polytopes. :math:`x \in \mathbb{R}^n` is the vector of optimization variables, where :math:`n = IK`. 

.. _probsolution :

Problem Solution
----------------

We briefly outline the solution mechanism here. For more details please see `Basu et. al (2020)
<http://proceedings.mlr.press/v119/basu20a/basu20a.pdf>`_.
To solve the problem we introduce the perturbed problem

.. math::
  \begin{array}{ll}
    \mbox{minimize} & x^T c  + \frac{\gamma}{2}x^T x \\
    \mbox{subject to} & A x \leq b \\
    & x_i \in \mathcal{C}_i \;\; \text{for all}\; = 1,\ldots, I
  \end{array}

where :math:`\gamma > 0` controls the tradeoff of between problem approximation and the smoothness of the dual objective function.
To make the above problem amenable to first order methods, we consider the Lagrangian dual,

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
For more details, we refer to `Basu et. al (2020)
<http://proceedings.mlr.press/v119/basu20a/basu20a.pdf>`_.


.. _algorithm :

The Algorithm
-------------

The overall algorithm can now be written as:

1. Start with an initial :math:`\lambda`
2. Get Primal: :math:`x_{\gamma}^*(\lambda)`
3. Get Gradient: :math:`Ax_{\gamma}^*(\lambda) - b`
4. Update :math:`\lambda` via appropriate mechanisms.
5. Continue till converge.
   
We currently support three different mechanisms for doing this first-order optimization. Specifically, `Proximal Gradient Ascent
<https://en.wikipedia.org/wiki/Proximal_gradient_method>`_, `Accelerated Gradient Ascent
<https://www.ceremade.dauphine.fr/~carlier/FISTA>`_, and `LBFGS-B
<https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_. For the details please see Appendix A of the `full paper
<https://arxiv.org/abs/2103.05277>`_.

.. _constraints :

Constraint Sets :math:`\mathcal{C}_i`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this current version of the solver we support a wide variety of constraints types :math:`\mathcal{C}_i`, 
such as:

1. Unit Box: :math:`\mathcal{C}_i = \big\{ x \in \mathbb{R}^K : 0 <= x_k <= 1\big\}`
2. Simplex-E: :math:`\mathcal{C}_i = \big\{ x \in \mathbb{R}^K : x_1 + ... + x_K = 1, \;\; x_k \geq 0\big\}`
3. Simplex-I: :math:`\mathcal{C}_i = \big\{ x \in \mathbb{R}^K : x_1 + ... + x_K \leq 1, \;\; x_k \geq 0\big\}`
4. Box Cut-E: :math:`\mathcal{C}_i = \big\{ x \in \mathbb{R}^K : x_1 + ... + x_K = d, \;\; 0 \leq x_k \leq 1\big\}`
5. Box Cut-I: :math:`\mathcal{C}_i = \big\{ x \in \mathbb{R}^K : x_1 + ... + x_K \leq d, \;\; 0 \leq x_k \leq 1\big\}`

Here :math:`E` and :math:`I` stands for equality and inequality. Also note that choosing :math:`d=1` the Box Cut reduces to the Simplex case. 


To execute step 2 of the overall algorithm, we need a projection operation on these constraint sets.
In our solver, we have implemented highly efficient projection algorithms to make step 2 extremely fast. The different sets have 
different customized algorithms to make the overall system highly efficient. For more details on 
these projection algorithms please see Section 3 of `the full paper
<https://arxiv.org/abs/2103.05277>`_.

.. _adaptive_smoothing :

Adaptive Smoothing Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The smoothness of :math:`g_\gamma` decreases as the number of constraints increases. 
A small :math:`\gamma` makes the optimizer's convergence prohibitively slow, while a large :math:`\gamma` reduces the accuracy of 
the solution. We define a practical criterion for sufficient convergence for a given :math:`\gamma` and 
implement a stage-wise algorithm that automatically reduces :math:`\gamma` when the criterion is met to 
prefer more accurate solutions. For details, please see Section 4 of the `full paper
<https://arxiv.org/abs/2103.05277>`_.

.. _convergence :

Stopping Criteria
^^^^^^^^^^^^^^^^^

Let :math:`\lambda_\gamma = \arg \max_{\lambda\ge 0} g_\gamma(\lambda)` and
:math:`\tilde{\lambda}_\gamma` be an approximate solution after the optimizer has made sufficient progress to maximize :math:`g_\gamma`.
If the approximation error :math:`(g_0(\lambda_0) - g_0(\tilde{\lambda}_\gamma))` is :math:`\epsilon` times smaller than the
total opportunity :math:`(g_0(\lambda_0) - g_0(0))` then we declare sufficient convergence, i.e.,

.. math::
    g_0(\lambda_0) - g_0(\tilde{\lambda}_\gamma) \le \epsilon \; (g_0(\lambda_0) - g_0(0)).

The intuition behind this is as follows:

#. The criterion is defined in terms of :math:`g_0` because it is the Lagrangian dual corresponding to the actual LP we want to solve and by strong duality, :math:`g_0(\lambda_0)` is the optimal primal objective that can be attained.
#. Since :math:`\lambda=0` removes the effect of constraints on the Lagrangian, :math:`g_0(0)` represents the maximum value of the primal objective. The total opportunity represents the value of objective "lost" to enforce the constraints :math:`Ax \le b`.
#. The approximation error (the left hand side of above) is due to two levels of approximation: (a) the error due to working with :math:`\gamma >0`, i.e., the difference between :math:`\lambda_0` and :math:`\lambda_\gamma`; and (b) the approximate solution of :math:`\max_\lambda g_\gamma(\lambda)`, i.e., the difference between :math:`\lambda_\gamma` and :math:`\tilde{\lambda}_\gamma`.



Infeasible problems
-------------------

Dualip is able to detect if the problem is primal infeasible. If the primal problem is infeasible,

.. math::
    g_\gamma^* = \max_{\lambda\ge 0} g_\gamma(\lambda) = \infty.

Furthermore, for any feasible :math:`x`, by weak duality, we have

.. math::
    g_\gamma^* & \leq \max_{x \in \mathcal{C} \; \text{and} \; x: Ax \leq b} ( c^T x + \frac{\gamma}{2} x^T x) \leq \max_{x \in \mathcal{C}} ( c^T x + \frac{\gamma}{2} x^T x) \\
    & = \sum_{i = 1}^I \max_{x_i\in\mathcal{C}_i} \; ({c_i}^T x_i + \frac{\gamma}{2} {x_i}^T x_i)

where the second inequality follows from the fact that the max is taken over a larger set. Now, for each constraint type :math:`\mathcal{C}_i`, it is easy to calculate a bound :math:`B` such that

.. math::
    \max_{x_i\in\mathcal{C}_i} \; ({c_i}^T x_i + \frac{\gamma}{2} {x_i}^T x_i) \leq B. 

If the primal is feasible, then strong optimality implies that :math:`{g_\gamma}^* \le IB`.
Thus, if, during the optimization, :math:`g_\gamma > IB`, then it guarantees that the primal is infeasible.


