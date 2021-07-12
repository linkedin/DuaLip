A Multi-Objective Optimization Problem
============================================
Multi-objective optimization problems (MOO) are optimization problems where more than one objective function needs to be optimized simultaneously. MOO appears frequently in machine learning and web applications where trade-offs between conflicting objective are needed. For example, balancing multiple business metrics in a ranking or recommendation system. MOO can be framed by considering one of the objectives as primary and others as constraints.

An example problem
------------------
We consider `volume optimization <https://www.kdd.org/kdd2016/papers/files/adf0710-guptaA.pdf>`_ as an example to demonstrate how to solve a Multi-Objective Optimization Problem. 
A core problem in any internet industry is to send out emails and notifications to the users,
either for marketing purposes, for gaining user attention, or to bring users back to the platform. They help in the
quick distribution of information, but too many emails or notifications are not preferable as they may lead bad user
experience.

The problem to be solved can be stated as: given a set of email/notificaiton, find a strategy of sending/dropping to maximize the overall session, 
such that the total number of emails or notifications that are sent is bounded, the overall click rate is above a threshold and the total disable rate is bounded as well.

How to translate the problem mathematically?
--------------------------------------------
To translate this problem mathematically, let us begin with some notations. Define

* :math:`x_{ik}`: Probability of sending the j-th email or notification to the i-th user.
* :math:`c_{ik}`: Probability of user visit the site if this email/notification is sent.
* :math:`p_{ik}`: Probability of user click this email/notification.
* :math:`r_{ik}`: Probability of user disable/unsubscribe this email/notification.

The optimization problem can be written as:

.. math::
  \begin{array}{ll}
    \mbox{Maximize} & \sum_{ik} x_{ik} c_{ik} &\\
    \mbox{subject to} & \sum_{ik} x_{ik} \leq c_1 & \text{Sends are bounded} \\
    & \sum_{ik} x_{ik} p_{ik} \geq c_2 & \text{Clicks above a threshold} \\
    & \sum_{ik} x_{ik} r_{ik} \leq c_3 & \text{Disables below a threshold} \\
    & 0 \leq x \leq 1 & \text{Probability constraint} \\
  \end{array}

We can further frame this using vector and matrix notation. 

#. The constraint matrix is

    .. math::
         A = \begin{bmatrix}
                1 & \ldots & 1 & \ldots & 1 & \ldots & 1\\
                -p_{11} & \ldots & -p_{1K} & \ldots & -p_{I1} & \ldots & -p_{IK}\\
                r_{11}  & \ldots & r_{1K} & \ldots & r_{I1} & \ldots & r_{IK}
            \end{bmatrix}

   **Note**: There is a sign change for the second constrain to get the :math:`\leq` form.
#. The constraint vector :math:`b` is :math:`\\(c_1, -c_2, c_3)`.
#. The objective vector :math:`c` is vectorized version of :math:`c_{ik}`.

By changing the maximization to a minimization problem, this now become the standard LP format:

.. math::
  \begin{array}{ll}
    \mbox{Minimize} & - x^T c \\
    \mbox{subject to} & Ax \leq b \\
    & x_i \in \mathcal{C}_i \;\; \text{for all}\; i
  \end{array}

where :math:`\mathcal{C}_i` is the unit box.


How to formulate the training data?
-----------------------------------
Let's consider a simple dataset. We have 3 emails, each with their predicted utilities. We would like to send less or equal to 2 emails. Keep click >= 1 and disable <= 0.03.

========= =========  ========  ==========
Email     Psession   Pclick    PDisable       
========= =========  ========  ==========
1         0.3        0.4       0.02
2         0.5        0.6       0.01
3         0.1        0.1       0.03
========= =========  ========  ==========

The matrix A will be 

.. math::
  \begin{bmatrix}
    & 1    &\; 1    &\; 1 \\
    & -0.4  &\; -0.6  &\; -0.1 \\
    & 0.02 &\; 0.01 &\; 0.03 \\
  \end{bmatrix}

The vector b will be (2, -1, 0.03). Vector c will be (-0.3, -0.5, -0.1).
The solver takes ACBlock and vectorB as two inputs. We require the input to be in the :ref:`Input Format`.

ACBlock:

.. code:: json

	{"id":1,"a":[[1.0],[-0.4],[0.02]],"c":[-0.3]}
	{"id":2,"a":[[1.0],[-0.6],[0.01]],"c":[-0.5]}
	{"id":3,"a":[[1.0],[-0.1],[0.03]],"c":[-0.1]}

vectorB:

.. code:: json

	{"row":1,"value":2.0}
	{"row":2,"value":-1.0}
	{"row":3,"value":0.03}

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
The solver can be run locally with spark-submit:

.. code:: bash

	$SPARK_HOME/bin/spark-submit --packages org.apache.spark:spark-avro_2.11:2.4.0 \
  --class com.linkedin.dualip.solver.LPSolverDriver ./dualip/build/libs/dualip_2.11.jar \
	--driver.objectiveClass com.linkedin.dualip.problem.MooSolverDualObjectiveFunction \
	--driver.solverOutputPath output/moo/ \
	--driver.gamma 1E-6 \
	--driver.outputFormat json \
	--driver.projectionType simplex \
	--input.ACblocksPath data/moo/data.json \
	--input.vectorBPath data/moo/budget.json \
	--input.metadataPath data/moo/metaData \
	--input.format json \
	--optimizer.solverType LBFGSB \
	--optimizer.dualTolerance 1E-8 \
	--optimizer.slackTolerance 5E-6 \
	--optimizer.maxIter 100 


How to read the results and do inference?
-----------------------------------------
There are two scenarios when reading the results. We can directly use the primal as decision variables. This is useful for a static system or batch processing.
Or we can use the dual to recover primal. This is useful when the system is dynamic and there are new items coming in. We can get the primal decision variable
:math:`x_{ij}` without even solving the optimization problem. This allows us to work in a low-latency environment as required by most internet applications.

The mechanism of solving such problems in industry is to first solve an extreme-scale problem to generate the duals and then use the duals in a low-latency environment to recover the primal, without the need of solving any optimization problem for every new item that is coming into
the ecosystem.

.. note::
	The above method for re-using the dual variable works as long as the score distribution of the new items
	matches that of the old items which were used to solve the Problem. To prevent staleness, in practice, the optimization problem is solved at a regular cadence.

