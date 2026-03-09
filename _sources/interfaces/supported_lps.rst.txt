.. _supported_lps:

Supported LPs
==============
DuaLip can be used to solve a wide variety of linear programs arising in web applications, including item matching, ranking, and related optimization problems. 
Motivated by the breadth of real-world use cases that can be formulated as **matching problems**, DuaLip provides efficient and scalable implementations tailored specifically to this class of workloads. 

The canonical matching problem aims to assign item :math:`j` to user :math:`i` with probability
:math:`x_{ij}` in order to maximize the total return, revenue, or profit.

.. math::

   \begin{array}{ll}
     \text{minimize}   & c^\top x \\
     \text{subject to} & A x \le b \\
                       & x_i \in \mathcal{C}_i \quad \text{for all } i = 1, \ldots, I
   \end{array}

We assume that the constraint matrix :math:`A` has a block-diagonal structure,
:math:`A = \mathrm{diag}(A_1, \ldots, A_I)`, where each block
:math:`A_i \in \mathbb{R}^{K \times K}`. These diagonal blocks are typically very sparse.
The constraint vector :math:`b` is dense and usually encodes capacity or budget limits,
such as the maximum number of users that can be matched to a given item.

This structured and unstructured sparsity of :math:`A` and :math:`C` enables efficient parallelization over GPU hardware, significantly reducing memory footprint and improving computational efficiency at scale. 
Projection operations can also be batched to fully utilize GPU throughput.

Note that in the case of a matching problem, parallelism is already supported and triggered when choosing the number of devices to exceed one. 
However, in the case of custom objective functions, the user needs to implement the parallelism themselves.

.. _probsolution:

Finally, the base matching problem formulation can be naturally extended to accommodate more complex constraints and objective functions. 
See the :ref:`A Matching Problem with Fairness Constraints <matching_complex>` section for more details.

