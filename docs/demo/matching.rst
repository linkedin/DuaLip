A Matching Problem
========================
Large-scale matching problems arise naturally in two-sided marketplaces with creators and consumers, where each item is associated with a budget and the objective is to maximize utility subject to budget constraints. 
Examples include ad marketplaces, where impressions must be allocated across campaigns without exceeding campaign budgets, and job marketplaces, where paid job impressions are distributed to maximize applies under similar constraints. 
We use a simple example to illustrate how such matching problems can be formulated and solved using DuaLip.


An example problem: Movie recommendations
-----------------------------------------
One example of a matching problem is movie recommendations. We want to recommend movies to users in a way that maximizes the total expected ratings, but without recommending any one movie too many times. 
We will use the `"MovieLens dataset" <https://grouplens.org/datasets/movielens/>`_ (`Harper et. al. 2015
<https://dl.acm.org/doi/10.1145^2/2827872>`_), which contains user rating of movies, for this example. (Similar problems have been framed in `Makari et. al. (2013) <https://dl.acm.org/doi/10.14778^2/2536360.2536362>`_.

How to frame the problem mathematically?
----------------------------------------
To frame this problem mathematically, define the following quantities:

* :math:`x_{ik}`: Probability of recommmending movie :math:`k` to user :math:`i`.
* :math:`c_{ik}`: Movie rating of movie :math:`k` by user :math:`i`.
* :math:`b_{k}`: Maximum number of times movie :math:`k` can be recommended.

The optimization problem can be written as:

.. math::
  \begin{array}{ll}
    \mbox{Maximize} & \sum_{i,k} x_{ik} c_{ik} \\
    \mbox{subject to} & \sum_i x_{ik} \leq b_k \;\; \text{for all}\;\; k = 1,\ldots, K \\
    & \sum_{k} x_{ik} \leq 1, \;\; \text{and} \;\; 0 \leq x_{i,k} \leq 1 \;\; \text{for all}\; i,k
  \end{array}

We can further frame this in the vector matrix notation by writing :math:`x,b,c` as the vectorized versions of :math:`x_{ik},b_k,c_{ik}`
respectively, e.g., :math:`x_i = (x_{i,1}, \ldots, x_{i,K})`. With this notation and changing the maximization to a
minimization problem, we have

.. math::
  \begin{array}{ll}
    \mbox{Minimize} & - x^T c \\
    \mbox{subject to} & Ax \leq b \\
    & x_i \in \mathcal{C}_i \;\; \text{for all}\; i
  \end{array}

where :math:`\mathcal{C}_i` is the closed simplex defined as

.. math::
    \mathcal{C}_i = \big\{ x \in \mathbb{R}^K : x_1 + ... + x_K \leq 1, \;\; x_k \geq 0\big\}

and :math:`A_{K \times IK}` is the matrix that encodes the :math:`K` movie limit constraints.

How to formulate the training data?
-----------------------------------
Let's consider a simple dataset. We have 4 users and 3 movies, each with their given rating.

========  =========  ========
UserID    MovieId    Rating
========  =========  ========
1         1          3
1         2          4
1         3          NA
2         1          NA
2         2          1
2         3          2
3         1          NA
3         2          2
3         3          1
4         1          2
4         2          4
4         3          3
========  =========  ========

The matrix A (dense) will be

.. math::
  A =
  \begin{bmatrix}
    1 &0 &0 &0 &0 &0 &0 &0 &0 &1 &0 &0 \\
    0 &1 &0 &0 &1 &0 &0 &1 &0 &0 &1 &0 \\
    0 &0 &0 &0 &0 &1 &0 &0 &1 &0 &0 &1 \\
  \end{bmatrix}

and the matrix C (dense) will be

.. math::
  C =
  \begin{bmatrix}
    3 &0 &0   &0 &0 &0   &0 &0 &0    &2 &0 &0 \\
    0 &4 &0   &0 &1 &1   &0 &2 &0    &0 &4 &0 \\
    0 &0 &0   &0 &0 &2   &2 &0 &1    &0 &0 &3 \\
  \end{bmatrix}

.. note::
  Note that in many matching problems, the connections are very sparse, i.e. users only rate a small percentage of movies. 
  If user i does not rate movie k (corresponding to NAs in the "Rating" column), the corresponding entry in A should be 0 because it will not contribute to the objective value. 
  Thus, the input tensor A is constructed by only including the non-zero entries.
  We thus store A and C in CSC sparse format that saves memory and improves computational efficiency by allowing efficient access to columns of A and C.

The vector b (dense) will be (1, 1, 1).

How to execute the solver?
--------------------------
To run the solver, ensure that PyTorch with CUDA support is installed and properly configured.
Then, follow the instructions to install the solver. See :ref:`Installation <installation>` for more details.

Get the dataset
^^^^^^^^^^^^^^^^^^^^^^^^^
Run the code below to download the 20M `MovieLens dataset <https://grouplens.org/datasets/movielens/>`_:

.. code:: bash

  curl -O https://files.grouplens.org/datasets/movielens/ml-20m.zip
  unzip ml-20m.zip
  mkdir data/ml-20m/data
  mv ml-20m/ratings.csv data/ml-20m/data

Run the solver
^^^^^^^^^^^^^^^^^^^^^^^^^
The solver can be run locally with the following command:

.. code:: bash

  python demo/movies_lens_matching.py --out_prefix data/movielens/ --run_solver

This will save the input tensors and projection maps to the directory specified by :code:`--out_prefix`.
It will also run the solver and save the results to the directory specified by :code:`--out_prefix`.

The solver will output the following information:

* The dual objective value.
* The shape of the input tensors.
* The shape of the projection maps.


How to interpret the results?
------------------------
We can see that as the solver progresses, :code:`dual_obj` increases while :code:`max_pos_slack` and :code:`max_zero_slack` decrease.

.. code:: text

	------------------------------------------------------------------------
	             Dual Decomposition based Linear Program Solver
	------------------------------------------------------------------------

	Optimizer: AGD solver
	primalUpperBound: -1.64155850e+05, maxIter: 500, dualTolerance: 1.0E-8 slackTolerance: 5.0E-6

	iter:     0	dual_obj: -6.86709416e+05	cx: -6.87890000e+05	feasibility: 9.650435e+01	λ(Ax-b): 0.000000e+00	γ||x||^2/2: 1.180584e+03	max_pos_slack: -Infinity	max_zero_slack: 9.650435e+01	abs_slack_sum: 8.828194e+04	time(sec): 8.987
	iter:     1	dual_obj: -6.86709416e+05	cx: -6.87890000e+05	feasibility: 9.650435e+01	λ(Ax-b): 0.000000e+00	γ||x||^2/2: 1.180584e+03	max_pos_slack: -Infinity	max_zero_slack: 9.650435e+01	abs_slack_sum: 8.828194e+04	time(sec): 6.738
	iter:     2	dual_obj: -3.26194182e+06	cx: -6.19067400e+05	feasibility: 3.907679e+01	λ(Ax-b): -2.646088e+06	γ||x||^2/2: 3.213506e+03	max_pos_slack: 1.680045e+01	max_zero_slack: 3.907679e+01	abs_slack_sum: 6.284022e+04	time(sec): 4.228
	iter:     3	dual_obj: -1.44149679e+06	cx: -6.24132300e+05	feasibility: 3.448183e+01	λ(Ax-b): -8.206471e+05	γ||x||^2/2: 3.282595e+03	max_pos_slack: 2.642451e+01	max_zero_slack: 3.448183e+01	abs_slack_sum: 6.233273e+04	time(sec): 4.558
	iter:     4	dual_obj: -8.82622744e+05	cx: -6.39467479e+05	feasibility: 3.030013e+01	λ(Ax-b): -2.466456e+05	γ||x||^2/2: 3.490308e+03	max_pos_slack: 2.471272e+01	max_zero_slack: 3.030013e+01	abs_slack_sum: 6.188493e+04	time(sec): 4.206
	iter:     5	dual_obj: -7.22368086e+05	cx: -6.58265033e+05	feasibility: 2.397677e+01	λ(Ax-b): -6.780186e+04	γ||x||^2/2: 3.698804e+03	max_pos_slack: 2.397677e+01	max_zero_slack: 2.053982e+01	abs_slack_sum: 6.032372e+04	time(sec): 4.321
	iter:     6	dual_obj: -6.82631528e+05	cx: -6.72533898e+05	feasibility: 1.847038e+01	λ(Ax-b): -1.381724e+04	γ||x||^2/2: 3.719608e+03	max_pos_slack: 1.847038e+01	max_zero_slack: 8.919135e+00	abs_slack_sum: 5.726833e+04	time(sec): 4.120
	iter:     7	dual_obj: -6.82631528e+05	cx: -6.72533898e+05	feasibility: 1.847038e+01	λ(Ax-b): -1.381724e+04	γ||x||^2/2: 3.719608e+03	max_pos_slack: 1.847038e+01	max_zero_slack: 8.919135e+00	abs_slack_sum: 5.726833e+04	time(sec): 4.442
	iter:     8	dual_obj: -6.82631528e+05	cx: -6.72533898e+05	feasibility: 1.847038e+01	λ(Ax-b): -1.381724e+04	γ||x||^2/2: 3.719608e+03	max_pos_slack: 1.847038e+01	max_zero_slack: 8.919135e+00	abs_slack_sum: 5.726833e+04	time(sec): 4.235
	iter:     9	dual_obj: -6.70900248e+05	cx: -6.73704610e+05	feasibility: 1.584054e+01	λ(Ax-b): -1.984704e+03	γ||x||^2/2: 4.789067e+03	max_pos_slack: 1.584054e+01	max_zero_slack: 1.676255e+00	abs_slack_sum: 4.087020e+04	time(sec): 3.694
	iter:    10	dual_obj: -6.70900248e+05	cx: -6.73704610e+05	feasibility: 1.584054e+01	λ(Ax-b): -1.984704e+03	γ||x||^2/2: 4.789067e+03	max_pos_slack: 1.584054e+01	max_zero_slack: 1.676255e+00	abs_slack_sum: 4.087020e+04	time(sec): 3.596


The solver achieves a dual objective value of Y and a primal objective value of Z after X iterations.
The primal-dual gap is W in the quadratic approximation of the objective function.
The maximum positive slack is X and the absolute sum of the slack is Z.
The time taken for the solver to run is A seconds.

.. code:: text

	iter:  1290	dual_obj: -6.28011839e+05	cx: -6.32627842e+05	feasibility: 1.289256e-02	λ(Ax-b): -5.876918e+00	γ||x||^2/2: 4.621880e+03	max_pos_slack: 1.813359e-02	max_zero_slack: 0.000000e+00	abs_slack_sum: 1.094879e+01	time(sec): 4.170
	iter:  1291	dual_obj: -6.28011839e+05	cx: -6.32660904e+05	feasibility: 9.317112e-02	λ(Ax-b): 2.721294e+01	γ||x||^2/2: 4.621852e+03	max_pos_slack: 9.317112e-02	max_zero_slack: 0.000000e+00	abs_slack_sum: 1.918328e+01	time(sec): 3.758
	iter:  1292	dual_obj: -6.28011839e+05	cx: -6.32660904e+05	feasibility: 9.317112e-02	λ(Ax-b): 2.721294e+01	γ||x||^2/2: 4.621852e+03	max_pos_slack: 9.317112e-02	max_zero_slack: 0.000000e+00	abs_slack_sum: 1.918328e+01	time(sec): 3.855
	Primal: -628012.671815458
	Dual: -628011.8409963399

How to do inference?
--------------------
There are two scenarios when reading the results. First, we can use the optimal primal values directly as decision variables. This is useful for a static system or batch processing.

Second, we can use the optimal dual values to recover primal. This is useful when the system is dynamic and there are new items coming in. We can get the primal decision variable
:math:`x_{ij}` without solving the extreme-scale optimization problem again. This allows us to work in a low-latency environment as required by most internet applications.

.. note::
	The above method for re-using the dual variable works as long as the score distribution of the new items
	matches that of the old items which were used to solve the MOO problem. To prevent staleness in practice, the optimization problem is re-solved at a regular cadence.
