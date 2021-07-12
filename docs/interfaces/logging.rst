.. _logging :

Logging
=======

Pre-Execution Logging
---------------------

The solver logs all the parameter information before execution:

.. code:: text

	------------------------------------------------------------------------
	                          Dualip v1.0     2021
	             Dual Decomposition based Linear Program Solver
	------------------------------------------------------------------------
	Settings:
	--driver.objectiveClass com.linkedin.dualip.problem.MooSolverDualObjectiveFunction
	--driver.solverOutputPath /dualip/test
	--driver.gamma 1E-6
	--driver.outputFormat avro
	--driver.projectionType simplex
	--driver.verbosity 1
	--input.ACblocksPath /unit-test/moo/60/data
	--input.vectorBPath /unit-test/moo/60/constraint
	--input.metadataPath /unit-test/moo/60/metaData
	--input.format avro
	--optimizer.solverType LBFGSB
	--optimizer.dualTolerance 1E-8
	--optimizer.slackTolerance 5E-6
	--optimizer.maxIter 100

In-Execution Logging
--------------------
The solver prints the following information during execution:

================================  ====================================================================================
Variables                         Description
================================  ====================================================================================
:code:`iter`                      Iteration number
:code:`dual_obj`                  Dual objective
:code:`max_pos_slack`             If :math:`\lambda_j` is not 0, :code:`max_pos_slack` = :math:`\max \{ (Ax-b)_j, 0 \} / (1 + |b_j|)`
:code:`max_zero_slack`            If :math:`\lambda_j` is 0, :code:`max_zero_slack` = :math:`\max\{ (Ax-b)_j, 0\} / (1 + |b_j|)`
:code:`abs_slack_sum`             Sum of violation
:code:`feasibility`               :math:`\max \{ r_j/(1 + |b_j|))\}`
:code:`cx`                        :math:`c^T x`
:math:`\lambda(Ax-b)`             Gradient
:math:`\frac{\gamma}{2}||x||^2`   Regularization term
:code:`time`                      Execution time of this iteration in second
================================  ====================================================================================

Termination Logging
-------------------

The solver prints one of the four possible results at termination:

================================  ====================================================================================
Status                            Description
================================  ====================================================================================
:code:`Converged`                 The Algorithm has converged according to the convergence criteria
:code:`Infeasible`                This happens when the dual objective has exceeded the primal upper bound.
:code:`Terminated`                The solver has reached maximum number of iteration. Users can look at the log to determine if the results are good enough to be used.
:code:`Failed`                    The solver failed during execution. Specific failer reason will be given in logs.
================================  ====================================================================================

We also show the number of iterations, the final primal and dual objective values and the number of active constraints.

Sample log
-----------------

.. code:: text

	------------------------------------------------------------------------
	                          Dualip v1.0     2021
	             Dual Decomposition based Linear Program Solver
	------------------------------------------------------------------------
	Settings:
	--driver.objectiveClass com.linkedin.dualip.problem.MooSolverDualObjectiveFunction
	--driver.solverOutputPath /dualip/test
	--driver.gamma 1E-6
	--driver.outputFormat avro
	--driver.projectionType simplex
	--driver.verbosity 1
	--input.ACblocksPath /unit-test/moo/60/data
	--input.vectorBPath /unit-test/moo/60/constraint
	--input.metadataPath /unit-test/moo/60/metaData
	--input.format avro
	--optimizer.solverType LBFGSB
	--optimizer.dualTolerance 1E-8
	--optimizer.slackTolerance 5E-6
	--optimizer.maxIter 100

	Optimizer: LBFGSB solver
	primalUpperBound: 2.50000000e-05, maxIter: 100, dualTolerance: 1.0E-8 slackTolerance: 5.0E-6

	iter:     0	dual_obj: -4.99999750e+01	cx: -5.00000000e+01	feasibility: 2.217486e+01	λ(Ax-b): 0.000000e+00	γ||x||/2: 2.500000e-05	max_pos_slack: -Infinity	max_zero_slack: 2.217486e+01	abs_slack_sum: 3.146621e+01	time(sec): 2.768
	iter:     1	dual_obj: -4.99999750e+01	cx: -5.00000000e+01	feasibility: 2.217486e+01	λ(Ax-b): 0.000000e+00	γ||x||/2: 2.500000e-05	max_pos_slack: -Infinity	max_zero_slack: 2.217486e+01	abs_slack_sum: 3.146621e+01	time(sec): 0.701
	iter:     2	dual_obj: -1.36522019e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.265221e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.587
	iter:     3	dual_obj: -1.36522019e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.265221e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.766
	iter:     4	dual_obj: -1.36522019e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.265221e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.714
	iter:     5	dual_obj: -1.34925665e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.249257e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.640
	iter:     6	dual_obj: -1.34127487e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.241275e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.481
	iter:     7	dual_obj: -1.32930221e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.229303e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.511
	iter:     8	dual_obj: -1.31134322e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.211344e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.575
	iter:     9	dual_obj: -1.28440474e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.184405e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.569
	iter:    10	dual_obj: -1.24399701e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.143998e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.585
	iter:    11	dual_obj: -1.18338541e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.083386e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.563
	iter:    12	dual_obj: -1.09246802e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -9.924686e+00	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.562
	iter:    13	dual_obj: -9.56091935e+00	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -8.560925e+00	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.509
	iter:    14	dual_obj: -7.68762043e+00	cx: -2.00000000e+00	feasibility: -2.473638e-01	λ(Ax-b): -5.687626e+00	γ||x||/2: 5.800000e-06	max_pos_slack: 2.473638e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.475
	iter:    15	dual_obj: -7.68762043e+00	cx: -2.00000000e+00	feasibility: -2.473638e-01	λ(Ax-b): -5.687626e+00	γ||x||/2: 5.800000e-06	max_pos_slack: 2.473638e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.681
	iter:    16	dual_obj: -7.68762043e+00	cx: -2.00000000e+00	feasibility: -2.473638e-01	λ(Ax-b): -5.687626e+00	γ||x||/2: 5.800000e-06	max_pos_slack: 2.473638e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.482
	iter:    17	dual_obj: -4.99999750e+01	cx: -5.00000000e+01	feasibility: 2.217486e+01	λ(Ax-b): 0.000000e+00	γ||x||/2: 2.500000e-05	max_pos_slack: -Infinity	max_zero_slack: 2.217486e+01	abs_slack_sum: 3.146621e+01	time(sec): 0.529
	iter:    18	dual_obj: -5.59665447e+00	cx: -7.00000000e+00	feasibility: 1.518062e-01	λ(Ax-b): 1.403338e+00	γ||x||/2: 7.800000e-06	max_pos_slack: 1.518062e-01	max_zero_slack: -Infinity	abs_slack_sum: 2.154136e-01	time(sec): 0.553
	iter:    19	dual_obj: -5.59665447e+00	cx: -7.00000000e+00	feasibility: 1.518062e-01	λ(Ax-b): 1.403338e+00	γ||x||/2: 7.800000e-06	max_pos_slack: 1.518062e-01	max_zero_slack: -Infinity	abs_slack_sum: 2.154136e-01	time(sec): 0.630
	iter:    20	dual_obj: -5.59665447e+00	cx: -7.00000000e+00	feasibility: 1.518062e-01	λ(Ax-b): 1.403338e+00	γ||x||/2: 7.800000e-06	max_pos_slack: 1.518062e-01	max_zero_slack: -Infinity	abs_slack_sum: 2.154136e-01	time(sec): 0.511
	iter:    21	dual_obj: -5.84994265e+00	cx: -4.00000000e+00	feasibility: -1.278211e-01	λ(Ax-b): -1.849949e+00	γ||x||/2: 6.600000e-06	max_pos_slack: 1.278211e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.548
	iter:    22	dual_obj: -5.53061009e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.306171e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.557
	iter:    23	dual_obj: -5.53061009e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.306171e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.597
	iter:    24	dual_obj: -5.53061009e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.306171e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.631
	iter:    25	dual_obj: -5.51271230e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.127193e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.701
	iter:    26	dual_obj: -5.50376340e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.037704e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.574
	iter:    27	dual_obj: -5.50964555e+00	cx: -6.00000000e+00	feasibility: 4.952939e-02	λ(Ax-b): 4.903471e-01	γ||x||/2: 7.400000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 7.028239e-02	time(sec): 0.697
	iter:    28	dual_obj: -5.50037908e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.003861e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.580
	iter:    29	dual_obj: -5.50110192e+00	cx: -6.00000000e+00	feasibility: 4.952939e-02	λ(Ax-b): 4.988907e-01	γ||x||/2: 7.400000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 7.028239e-02	time(sec): 0.441
	iter:    30	dual_obj: -5.50002401e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.000310e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.536
	iter:    31	dual_obj: -5.50012824e+00	cx: -6.00000000e+00	feasibility: 4.952939e-02	λ(Ax-b): 4.998644e-01	γ||x||/2: 7.400000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 7.028239e-02	time(sec): 0.493
	iter:    32	dual_obj: -5.49999490e+00	cx: -6.00000000e+00	feasibility: 4.952939e-02	λ(Ax-b): 4.999977e-01	γ||x||/2: 7.400000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 7.028239e-02	time(sec): 0.451
	iter:    33	dual_obj: -5.49999588e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.000029e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.387
	iter:    34	dual_obj: -5.49999301e+00	cx: -5.82333310e+00	feasibility: 3.202898e-02	λ(Ax-b): 3.233328e-01	γ||x||/2: 7.242060e-06	max_pos_slack: 3.202898e-02	max_zero_slack: -Infinity	abs_slack_sum: 4.544925e-02	time(sec): 0.374
	iter:    35	dual_obj: -5.49999301e+00	cx: -5.82333310e+00	feasibility: 3.202898e-02	λ(Ax-b): 3.233328e-01	γ||x||/2: 7.242060e-06	max_pos_slack: 3.202898e-02	max_zero_slack: -Infinity	abs_slack_sum: 4.544925e-02	time(sec): 0.693
	iter:    36	dual_obj: -5.49999301e+00	cx: -5.82333310e+00	feasibility: 3.202898e-02	λ(Ax-b): 3.233328e-01	γ||x||/2: 7.242060e-06	max_pos_slack: 3.202898e-02	max_zero_slack: -Infinity	abs_slack_sum: 4.544925e-02	time(sec): 0.422
	iter:    37	dual_obj: -5.51201647e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.120235e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.477
	iter:    38	dual_obj: -5.50127583e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.012828e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.489
	iter:    39	dual_obj: -5.50012959e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.001366e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.608
	iter:    40	dual_obj: -5.50000727e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.000143e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.847
	iter:    41	dual_obj: -5.49999421e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.000012e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.541
	iter:    42	dual_obj: -5.49999295e+00	cx: -5.44549277e+00	feasibility: -5.399420e-03	λ(Ax-b): -5.450721e-02	γ||x||/2: 7.029980e-06	max_pos_slack: 5.399420e-03	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.537
	iter:    43	dual_obj: -5.49999295e+00	cx: -5.44549277e+00	feasibility: -5.399420e-03	λ(Ax-b): -5.450721e-02	γ||x||/2: 7.029980e-06	max_pos_slack: 5.399420e-03	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.523
	iter:    44	dual_obj: -5.49999295e+00	cx: -5.44549277e+00	feasibility: -5.399420e-03	λ(Ax-b): -5.450721e-02	γ||x||/2: 7.029980e-06	max_pos_slack: 5.399420e-03	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.548
	iter:    45	dual_obj: -5.49999295e+00	cx: -5.50000000e+00	feasibility: 9.492812e-13	λ(Ax-b): 9.583005e-12	γ||x||/2: 7.050000e-06	max_pos_slack: 9.492812e-13	max_zero_slack: -Infinity	abs_slack_sum: 1.347034e-12	time(sec): 0.580
	iter:    46	dual_obj: -5.49999295e+00	cx: -5.50000000e+00	feasibility: 9.492812e-13	λ(Ax-b): 9.583005e-12	γ||x||/2: 7.050000e-06	max_pos_slack: 9.492812e-13	max_zero_slack: -Infinity	abs_slack_sum: 1.347034e-12	time(sec): 0.570
	Total LBFGS iterations: 7
	Status:Converged
	Total number of iterations: 47
	Primal: -5.44548574018906
	Dual: -5.499992951782625
	Number of Active Constraints: 1
