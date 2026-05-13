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

  python examples/movielens_matching/movies_lens_matching.py --out_prefix data/movielens/ --run_solver --ratings_csv_path data/ml-20m/data/ratings.csv

This will save the input tensors and projection maps to the directory specified by :code:`--out_prefix`.
It will also run the solver and save the results to the directory specified by :code:`--out_prefix`.

The solver will print the following information:
* The objective result at each iteration.
* The final dual objective value.
* The shape of the input tensors.


How to interpret the results?
------------------------
We can see that as the solver progresses, :code:`dual_obj` increases while :code:`max_pos_slack` and :code:`max_zero_slack` decrease.

.. code:: text

  iter: 1    dual_objective: -686709.1875   dual_grad_norm: 8696.43359375     reg_penalty: 1180.583251953125    dual_val_times_grad: 0.0                  max_pos_slack: 2991.600341796875   sum_pos_slack: 88280.8671875
  iter: 2    dual_objective: -686591.625    dual_grad_norm: 8688.6513671875   reg_penalty: 1180.541259765625    dual_val_times_grad: 0.5346085429191589   max_pos_slack: 2983.8203125        sum_pos_slack: 88250.578125
  iter: 3    dual_objective: -686577.875    dual_grad_norm: 8679.375          reg_penalty: 1180.548583984375    dual_val_times_grad: 1.2171449661254883   max_pos_slack: 2974.052490234375   sum_pos_slack: 88231.953125
  iter: 4    dual_objective: -686572.75     dual_grad_norm: 8668.126953125    reg_penalty: 1180.55810546875     dual_val_times_grad: 2.0419816970825195   max_pos_slack: 2962.280029296875   sum_pos_slack: 88210.203125
  iter: 5    dual_objective: -686594.6875   dual_grad_norm: 8655.1201171875   reg_penalty: 1180.5743408203125   dual_val_times_grad: 3.0046846866607666   max_pos_slack: 2948.519775390625   sum_pos_slack: 88188.765625
  iter: 6    dual_objective: -686581.5      dual_grad_norm: 8640.044921875    reg_penalty: 1180.58251953125     dual_val_times_grad: 4.101046562194824    max_pos_slack: 2932.73046875       sum_pos_slack: 88158.2265625
  iter: 7    dual_objective: -686603.3125   dual_grad_norm: 8623.3701171875   reg_penalty: 1180.601806640625    dual_val_times_grad: 5.327372074127197    max_pos_slack: 2915.07739          sum_pos_slack: 88129.34375
  iter: 8    dual_objective: -686617.6875   dual_grad_norm: 8604.8583984375   reg_penalty: 1180.6212158203125   dual_val_times_grad: 6.679593086242676    max_pos_slack: 2895.616943359375   sum_pos_slack: 88095.171875
  iter: 9    dual_objective: -686609.0      dual_grad_norm: 8584.591796875    reg_penalty: 1180.648193359375    dual_val_times_grad: 8.153701782226562    max_pos_slack: 2874.402587890625   sum_pos_slack: 88054.6875
  iter: 10   dual_objective: -686623.4375   dual_grad_norm: 8562.7861328125   reg_penalty: 1180.6798095703125   dual_val_times_grad: 9.74583911895752     max_pos_slack: 2851.467041015625   sum_pos_slack: 88013.890625
  iter: 11   dual_objective: -686624.5625   dual_grad_norm: 8539.47265625     reg_penalty: 1180.7239990234375   dual_val_times_grad: 11.451858520507812   max_pos_slack: 2827.134765625      sum_pos_slack: 87968.296875

.. code:: text

  iter: 9998    dual_objective: -628012.8125   dual_grad_norm: 4406.33837890625   reg_penalty: 4623.5439453125    dual_val_times_grad: 16.16229248046875    max_pos_slack: 1.8836631774902344   sum_pos_slack: 171.60943603515625
  iter: 9999    dual_objective: -628012.875    dual_grad_norm: 4406.33837890625   reg_penalty: 4623.54638671875   dual_val_times_grad: 15.531607627868652   max_pos_slack: 1.88861083984375     sum_pos_slack: 171.57188415527344
  iter: 10000   dual_objective: -628012.875    dual_grad_norm: 4406.33837890625   reg_penalty: 4623.54931640625   dual_val_times_grad: 16.019495010375977   max_pos_slack: 1.8933353424072266   sum_pos_slack: 171.66769409179688

  Dual objective: -628012.875
  A shape: (26744, 138493) C shape: (26744, 138493) b shape: (26744,)

The solver achieves a dual objective value of -628012.875 after 10000 iterations.
The maximum positive slack is 1.8933353424072266 and the sum of positive slack is 171.66769409179688.

How to do inference?
--------------------
There are two scenarios when reading the results. First, we can use the optimal primal values directly as decision variables. This is useful for a static system or batch processing.

Second, we can use the optimal dual values to recover primal. This is useful when the system is dynamic and there are new items coming in. We can get the primal decision variable
:math:`x_{ij}` without solving the extreme-scale optimization problem again. This allows us to work in a low-latency environment as required by most internet applications.

.. note::
	The above method for re-using the dual variable works as long as the score distribution of the new items
	matches that of the old items which were used to solve the matching problem. To prevent staleness in practice, the optimization problem is re-solved at a regular cadence.
