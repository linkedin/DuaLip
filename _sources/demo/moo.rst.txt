A Multi-Objective Optimization Problem
============================================
Multi-objective optimization problems (MOO) are optimization problems where more than one objective function needs to be optimized simultaneously. MOO problems appear frequently in machine learning and web applications where we need to make trade-offs between conflicting objectives, e.g., balancing multiple business metrics in a ranking or recommendation system. We can formulate such problems as an optimization problem by considering one of the objectives as primary and the others as constraints.

An example problem: Email volume optimization
---------------------------------------------s
`Volume optimization <https://www.kdd.org/kdd2016/papers/files/adf0710-guptaA.pdf>`_ is an example of a MOO problem in the real world. 
A core problem for any internet company is to send out emails to the users. These could be sent for any number of reasons, e.g., marketing, gaining user attention, bringing users back to the platform. While emails distribute information quickly, too many emails may lead to a bad user experience.

The email optimization problem can be framed in the following way: given a set of emails and a set of users, determine which emails to send to which users to maximize the overall number of sessions, but such that:

#. The total number of emails that are sent is bounded,
#. The overall click rate is above a threshold, and
#. The overall disable rate is bounded above.

How to frame the problem mathematically?
----------------------------------------
To frame this problem mathematically, define the following quantites:

* :math:`x_{ik}`: Probability of sending the k-th email to the i-th user.
* :math:`c_{ik}`: Probability of i-th user visits the site if k-th email is sent.
* :math:`p_{ik}`: Probability of i-th user clicks k-th email.
* :math:`r_{ik}`: Probability of i-th user disables/unsubscribes k-th email.

The optimization problem can be written as:

.. math::
  \begin{array}{ll}
    \mbox{Maximize} & \sum_{ik} x_{ik} c_{ik} &\\
    \mbox{subject to} & \sum_{ik} x_{ik} \leq b_1 & \text{Sends are bounded} \\
    & \sum_{ik} x_{ik} p_{ik} \geq b_2 & \text{Clicks above a threshold} \\
    & \sum_{ik} x_{ik} r_{ik} \leq b_3 & \text{Disables below a threshold} \\
    & 0 \leq x \leq 1 & \text{Probability constraint} \\
  \end{array}

We can further frame this using vector and matrix notation:

#. The constraint matrix is

    .. math::
         A = \begin{bmatrix}
                1 & \ldots & 1 & \ldots & 1 & \ldots & 1\\
                -p_{11} & \ldots & -p_{1K} & \ldots & -p_{I1} & \ldots & -p_{IK}\\
                r_{11}  & \ldots & r_{1K} & \ldots & r_{I1} & \ldots & r_{IK}
            \end{bmatrix}

   **Note**: There is a sign change for the second constraint to get the :math:`\leq` form.
#. The constraint vector :math:`b` is :math:`\\(b_1, -b_2, b_3)`.
#. The objective vector :math:`c` is vectorized version of :math:`c_{ik}`.

By changing the maximization to a minimization problem, the problem is now in a standard LP format for our solver:

.. math::
  \begin{array}{ll}
    \mbox{Minimize} & - x^T c \\
    \mbox{subject to} & Ax \leq b \\
    & x_i \in \mathcal{C}_i \;\; \text{for all}\; i
  \end{array}

where :math:`\mathcal{C}_i` is the unit box.


How to formulate the training data?
-----------------------------------
Let's consider a simple dataset. We have 3 emails to be sent to a single user (i.e. :math:`i = 1` and :math:`k = 3`), each with their predicted utilities. We would like to send less or equal to 2 emails, have at least 1 or more clicks, and have the disable probability <= 0.03.

========= =========================  ========================  =========================
Email     Psession (:math:`c_{1k}`)   Pclick (:math:`p_{1k}`)  PDisable (:math:`r_{1k}`)      
========= =========================  ========================  =========================
1         0.3                        0.4                       0.02
2         0.5                        0.6                       0.01
3         0.1                        0.1                       0.03
========= =========================  ========================  =========================

The matrix A will be 

.. math::
  \begin{bmatrix}
    & 1    &\; 1    &\; 1 \\
    & -0.4  &\; -0.6  &\; -0.1 \\
    & 0.02 &\; 0.01 &\; 0.03 \\
  \end{bmatrix},

the vector :math:`b` will be :math:`(2, -1, 0.03)`, and the vector :math:`c` will be :math:`(-0.3, -0.5, -0.1)`.
The solver takes ACBlock and vectorB as two inputs. We require the input to be in the :ref:`Input Format`.

ACBlock:

.. code:: json

	{"id":1,"a":[[1.0],[-0.4],[0.02]],"c":[-0.3]}
	{"id":2,"a":[[1.0],[-0.6],[0.01]],"c":[-0.5]}
	{"id":3,"a":[[1.0],[-0.1],[0.03]],"c":[-0.1]}

Each line in the data above correponds to one email (indexed by `k`). Index `i` of :code:`a` in line `k` is the coefficient associated with email `k` in constraint `i`. :code:`c` in line `k` is the coefficient associated with email `k` in the objective function to be minimized.

vectorB:

.. code:: json

	{"row":1,"value":2.0}
	{"row":2,"value":-1.0}
	{"row":3,"value":0.03}

Each line in the data above correponds to one constraint. The :code:`value` is simply the right-hand side of that constraint.

How to execute the solver?
--------------------------
Here is a step-by-step tutorial on run a moo solver on your machine.

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
A sample unit test dataset of MOO has been prepared under directory data/moo. 

Run the solver
^^^^^^^^^^^^^^^^^^^^^^^^^
The solver can be run locally with :code:`spark-submit`. Below is an example of how to run the solver with specified parameters. (**Note:** The name of the :code:`.jar` file on your machine might be slightly different from :code:`./dualip/build/libs/dualip_2.12.jar` due to differing version numbers. Please replace the filename with the one on your machine if necessary.)

.. code:: bash

	$SPARK_HOME/bin/spark-submit --packages org.apache.spark:spark-avro_2.12:3.1.1 \
  --class com.linkedin.dualip.solver.LPSolverDriver ./dualip/build/libs/dualip_2.12.jar \
	--driver.objectiveClass com.linkedin.dualip.problem.MooSolverDualObjectiveFunction \
	--driver.solverOutputPath output/moo/ \
	--driver.gamma 1E-6 \
	--driver.outputFormat json \
	--driver.projectionType simplex \
	--input.ACblocksPath data/moo/data.json \
	--input.vectorBPath data/moo/budget.json \
	--input.format json \
	--optimizer.solverType LBFGSB \
	--optimizer.dualTolerance 1E-8 \
	--optimizer.slackTolerance 5E-6 \
	--optimizer.maxIter 100 


How to read the results and do inference?
-----------------------------------------
There are two scenarios when reading the results. First, we can use the optimal primal values directly as decision variables. This is useful for a static system or batch processing.

Second, we can use the optimal dual values to recover primal. This is useful when the system is dynamic and there are new items coming in. We can get the primal decision variable
:math:`x_{ij}` without solving the extreme-scale optimization problem again. This allows us to work in a low-latency environment as required by most internet applications.

.. note::
	The above method for re-using the dual variable works as long as the score distribution of the new items
	matches that of the old items which were used to solve the MOO problem. To prevent staleness in practice, the optimization problem is re-solved at a regular cadence.
