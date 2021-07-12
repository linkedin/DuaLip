A Matching Problem
========================
Large-scale matching problems occur naturally in many two-sided marketplaces. Usually in these settings, there are creators and consumers.
Each item created has an associated budget and the problem is to maximize the utility under budget constraints.
For example, in the ads marketplace, each ad campaign has a budget and we need to distribute impressions appropriately.
For the jobs marketplace, each paid job has a budget and the impressions need to be appropriately allocated to maximize job applies.
Here we work with a simple example to showcase how we can formulate and solve such matching problems through Dualip.

An example problem
------------------
For `"MovieLens dataset" <https://grouplens.org/datasets/movielens/>`_ (`Harper et. al. 2015
<https://dl.acm.org/doi/10.1145/2827872>`_), which contains user rating of movies, let us imagine, we want to recommend movies to users,
such that we maximize the total expected ratings, while restricting the number of times a movie is recommended. Similar
such problems have been framed in `Makari et. al. (2013)
<https://dl.acm.org/doi/10.14778/2536360.2536362>`_.

How to translate the problem mathematically?
--------------------------------------------
To translate this problem mathematically, let us begin with some notation. Define

* :math:`x_{ik}`: Probability of recommmending movie :math:`k` to user :math:`i`.
* :math:`c_{ik}`: Movie rating of movie :math:`k` by user :math:`i`.
* :math:`b_{k}`: Maximum number of times movie :math:`k` can be recommended.

Based on this notation, the above problem can be formulated mathematically as:

.. math::
  \begin{array}{ll}
    \mbox{Maximize} & \sum_{i,k} x_{i,k} c_{i,k} \\
    \mbox{subject to} & \sum_i x_{i,k} \leq b_k \;\; \text{for all}\;\; k = 1,\ldots, K \\
    & \sum_{k} x_{i,k} \leq 1, \;\; \text{and} \;\; 0 \leq x_{i,k} \leq 1 \;\; \text{for all}\; i,k
  \end{array}

We can further frame this in the vector matrix notation by writing :math:`x,b,c` as the vectorized version of :math:`x_{i,k},b_k,c_{i,k}`
respectively, with :math:`x_i = (x_{i,1}, \ldots, x_{i,K})`. Under this notation and changing the maximization to a
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

The matrix A will be

.. math::
  A =
  \begin{bmatrix}
    1 &0 &0 &0 &0 &0 &0 &0 &0 &1 &0 &0 \\
    0 &1 &0 &0 &1 &0 &0 &1 &0 &0 &1 &0 \\
    0 &0 &0 &0 &0 &1 &0 &0 &1 &0 &0 &1 \\
  \end{bmatrix}

.. note::
  In many matching problems, the connections are very sparse, i.e. users only rate a small percentage of movies. If user i does not rate movie k, the corresponding entry in A should be 0 because it will not contribute to the objective value. ACBlock only record the non-zero entries.

The vector b will be (1, 1, 1). Vector c will be (-3, -4, 0, 0, -1, -2, 0, -2, -1, -2, -4, -3)

The solver takes ACBlock and vectorB as two inputs. We require the input to be in the :ref:`Input Format`.

The format of ACBlock:

.. code:: json

  {
      "name" : "id",
      "type" : [ "string" ]
    }, {
      "name" : "data",
      "type" : [ {
        "type" : "array",
        "items" : [ {
          "type" : "record",
          "name" : "data",
          "fields" : [ {
            "name" : "rowId",
            "type" : "int"
          }, {
            "name" : "c",
            "type" : "double"
          }, {
            "name" : "a",
            "type" : "double"
          } ]
        } ]
      } ]
    }

Here id is a unique identifier of the block, i.e. user id for this problem. Each id correspond to an array of tuple, which is in the format of (rowId, c(rowId), a(rowId)). rowId correspond to movieId in this problem.

ACBlock in json format:

.. code:: json

  {"id":1,"data":[[1, -3, 1], [2, -4, 1]]}
  {"id":2,"data":[[2, -1, 1], [3, -2, 1]]}
  {"id":3,"data":[[2, -2, 1], [3, -1, 1]]}
  {"id":4,"data":[[1, -2, 1], [2, -4, 1], [3, -3, 1]]}

vectorB in json format:

.. code:: json

	{"row":1,"value":1.0}
	{"row":2,"value":1.0}
	{"row":3,"value":1.0}

How to execute the solver?
--------------------------
Here is a step-by-step tutorial on run a matching solver on your machine.

Install Spark
^^^^^^^^^^^^^^^^^^
This step is platform-dependent. On OS X, you can install Spark with Homebrew using the following command:

.. code:: bash

  brew install apache-spark

For more information, see the `Spark docs <http://spark.apache.org/docs/latest/index.html>`_.

Get and build the code
^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: bash

  ./gradlew build

Get the dataset
^^^^^^^^^^^^^^^^^^^^^^^^^
To download the 20M `MovieLens dataset <https://grouplens.org/datasets/movielens/>`_

.. code:: bash

  curl -O https://files.grouplens.org/datasets/movielens/ml-20m.zip
  unzip ml-20m.zip
  mkdir data
  mv rating.csv data/

Then use MatchingDataGenerator to convert the dataset to the format solver take: 

.. code:: bash

  $SPARK_HOME/bin/spark-submit --packages org.apache.spark:spark-avro_2.11:2.4.0 --class com.linkedin.dualip.preprocess.MatchingDataGenerator ./dualip/build/libs/dualip_2.11.jar \
  --preprocess.dataBasePath data/ml-20m/ \
  --preprocess.dataFormat csv \
  --preprocess.dataBlockDim userId \
  --preprocess.constraintDim movieId \
  --preprocess.budgetDim budget \
  --preprocess.budgetValue 30 \
  --preprocess.rewardDim rating \
  --preprocess.costGenerator constant \
  --preprocess.costValue 1.0 \
  --preprocess.outputPath data/movielens/

Here rewardDim is the column name corresponding to reward information :math:`c_{ij}`.
dataBlockDim is the column name corresponding to dataBlockId (i). constraintDim is the column name corresponding to itemId (j).
budgetValue is budget information :math:`b_{k}`.

Run the solver
^^^^^^^^^^^^^^^^^^^^^^^^^
The solver can be run locally with spark-submit:

.. code:: bash

  $SPARK_HOME/bin/spark-submit --packages org.apache.spark:spark-avro_2.11:2.4.0 \
  --class com.linkedin.dualip.solver.LPSolverDriver ./dualip/build/libs/dualip_2.11.jar \
  --driver.objectiveClass com.linkedin.dualip.problem.MatchingSolverDualObjectiveFunction \
  --driver.solverOutputPath /output/matching/ \
  --driver.gamma 0.1 \
  --driver.outputFormat json \
  --driver.projectionType simplexInequality \
  --input.ACblocksPath data/movielens/data \
  --input.vectorBPath data/movielens/budget \
  --input.metadataPath data/movielens/metaData \
  --input.format avro \
  --matching.slateSize 1 \
  --optimizer.solverType LBFGSB \
  --optimizer.dualTolerance 1E-8 \
  --optimizer.slackTolerance 5E-6 \
  --optimizer.maxIter 500 

How to read the results?
------------------------

We can see that as the solver progress, the :code:`dual_obj` increases and :code:`max_pos_slack` and :code:`max_zero_slack` decreases.

.. code:: text

	------------------------------------------------------------------------
	                          Dualip v1.0     2021
	             Dual Decomposition based Linear Program Solver
	------------------------------------------------------------------------

	Optimizer: LBFGSB solver
	primalUpperBound: -1.64155850e+05, maxIter: 500, dualTolerance: 1.0E-8 slackTolerance: 5.0E-6

	iter:     0	dual_obj: -6.86709416e+05	cx: -6.87890000e+05	feasibility: 9.650435e+01	λ(Ax-b): 0.000000e+00	γ||x||/2: 1.180584e+03	max_pos_slack: -Infinity	max_zero_slack: 9.650435e+01	abs_slack_sum: 8.828194e+04	time(sec): 8.987
	iter:     1	dual_obj: -6.86709416e+05	cx: -6.87890000e+05	feasibility: 9.650435e+01	λ(Ax-b): 0.000000e+00	γ||x||/2: 1.180584e+03	max_pos_slack: -Infinity	max_zero_slack: 9.650435e+01	abs_slack_sum: 8.828194e+04	time(sec): 6.738
	iter:     2	dual_obj: -3.26194182e+06	cx: -6.19067400e+05	feasibility: 3.907679e+01	λ(Ax-b): -2.646088e+06	γ||x||/2: 3.213506e+03	max_pos_slack: 1.680045e+01	max_zero_slack: 3.907679e+01	abs_slack_sum: 6.284022e+04	time(sec): 4.228
	iter:     3	dual_obj: -1.44149679e+06	cx: -6.24132300e+05	feasibility: 3.448183e+01	λ(Ax-b): -8.206471e+05	γ||x||/2: 3.282595e+03	max_pos_slack: 2.642451e+01	max_zero_slack: 3.448183e+01	abs_slack_sum: 6.233273e+04	time(sec): 4.558
	iter:     4	dual_obj: -8.82622744e+05	cx: -6.39467479e+05	feasibility: 3.030013e+01	λ(Ax-b): -2.466456e+05	γ||x||/2: 3.490308e+03	max_pos_slack: 2.471272e+01	max_zero_slack: 3.030013e+01	abs_slack_sum: 6.188493e+04	time(sec): 4.206
	iter:     5	dual_obj: -7.22368086e+05	cx: -6.58265033e+05	feasibility: 2.397677e+01	λ(Ax-b): -6.780186e+04	γ||x||/2: 3.698804e+03	max_pos_slack: 2.397677e+01	max_zero_slack: 2.053982e+01	abs_slack_sum: 6.032372e+04	time(sec): 4.321
	iter:     6	dual_obj: -6.82631528e+05	cx: -6.72533898e+05	feasibility: 1.847038e+01	λ(Ax-b): -1.381724e+04	γ||x||/2: 3.719608e+03	max_pos_slack: 1.847038e+01	max_zero_slack: 8.919135e+00	abs_slack_sum: 5.726833e+04	time(sec): 4.120
	iter:     7	dual_obj: -6.82631528e+05	cx: -6.72533898e+05	feasibility: 1.847038e+01	λ(Ax-b): -1.381724e+04	γ||x||/2: 3.719608e+03	max_pos_slack: 1.847038e+01	max_zero_slack: 8.919135e+00	abs_slack_sum: 5.726833e+04	time(sec): 4.442
	iter:     8	dual_obj: -6.82631528e+05	cx: -6.72533898e+05	feasibility: 1.847038e+01	λ(Ax-b): -1.381724e+04	γ||x||/2: 3.719608e+03	max_pos_slack: 1.847038e+01	max_zero_slack: 8.919135e+00	abs_slack_sum: 5.726833e+04	time(sec): 4.235
	iter:     9	dual_obj: -6.70900248e+05	cx: -6.73704610e+05	feasibility: 1.584054e+01	λ(Ax-b): -1.984704e+03	γ||x||/2: 4.789067e+03	max_pos_slack: 1.584054e+01	max_zero_slack: 1.676255e+00	abs_slack_sum: 4.087020e+04	time(sec): 3.694
	iter:    10	dual_obj: -6.70900248e+05	cx: -6.73704610e+05	feasibility: 1.584054e+01	λ(Ax-b): -1.984704e+03	γ||x||/2: 4.789067e+03	max_pos_slack: 1.584054e+01	max_zero_slack: 1.676255e+00	abs_slack_sum: 4.087020e+04	time(sec): 3.596


The solver converges after 413 iterations, while the combined number of iterations (including the
internal iterations of LBFGS) is 1293. We also show the final dual and primal objectives, as well
as the number of active constraints in the problem.

.. code:: text

	iter:  1290	dual_obj: -6.28011839e+05	cx: -6.32627842e+05	feasibility: 1.289256e-02	λ(Ax-b): -5.876918e+00	γ||x||/2: 4.621880e+03	max_pos_slack: 1.813359e-02	max_zero_slack: 0.000000e+00	abs_slack_sum: 1.094879e+01	time(sec): 4.170
	iter:  1291	dual_obj: -6.28011839e+05	cx: -6.32660904e+05	feasibility: 9.317112e-02	λ(Ax-b): 2.721294e+01	γ||x||/2: 4.621852e+03	max_pos_slack: 9.317112e-02	max_zero_slack: 0.000000e+00	abs_slack_sum: 1.918328e+01	time(sec): 3.758
	iter:  1292	dual_obj: -6.28011839e+05	cx: -6.32660904e+05	feasibility: 9.317112e-02	λ(Ax-b): 2.721294e+01	γ||x||/2: 4.621852e+03	max_pos_slack: 9.317112e-02	max_zero_slack: 0.000000e+00	abs_slack_sum: 1.918328e+01	time(sec): 3.855
	Total LBFGS iterations: 413
	Status:Converged
	Total number of iterations: 1293
	Primal: -628012.671815458
	Dual: -628011.8409963399
	Number of Active Constraints: 2992

The detailed log is given :ref:`here <Matching log>`.

How to do inference?
--------------------

There are two scenarios when reading the results. We can directly use the primal as decision variables. This is useful for a static system or batch processing.
Or we can use the dual to recover primal. This is useful when the system is dynamic and there are new items coming in. We can get the primal decision variable
:math:`x_{ij}` without even solving the optimization problem. This allows us to work in a low-latency environment as required by most internet applications.

The mechanism of solving such problems in industry is to first solve an extreme-scale problem to generate the duals and then use the duals in a
low-latency environment to recover the primal, without the need of solving any optimization problem for every new item that is coming into
the ecosystem.

.. note::
	The above method for re-using the dual variable works as long as the score distribution of the new items
	matches that of the old items which were used to solve the Problem. To prevent staleness, in practice, the optimization problem is solved at a regular cadence.
