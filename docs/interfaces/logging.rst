.. _logging :

Logging
=======

Pre-Execution Logging
---------------------

The solver logs all the parameter information before execution:

.. code:: text

	------------------------------------------------------------------------
	                          DuaLip v1.0     2021
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
:code:`gradientCall`              The number of calls to compute the gradient in the solver.
:code:`iter`                      Iteration number.
:code:`dual_obj`                  Dual objective value.
:code:`max_pos_slack`             If :math:`\lambda_j` is not 0, :code:`max_pos_slack` = :math:`\max \{ (Ax-b)_j, 0 \} / (1 + |b_j|)`.
:code:`max_zero_slack`            If :math:`\lambda_j` is 0, :code:`max_zero_slack` = :math:`\max\{ (Ax-b)_j, 0\} / (1 + |b_j|)`.
:code:`abs_slack_sum`             Sum of constraint violations.
:code:`feasibility`               :math:`\max \{ r_j/(1 + |b_j|)\}`.
:code:`cx`                        Primal objective value, i.e. :math:`c^T x`.
:math:`\lambda(Ax-b)`             Gradient.
:math:`\frac{\gamma}{2}||x||^2`   Regularization term.
:code:`time`                      Execution time of this iteration in seconds.
================================  ====================================================================================

Termination Logging
-------------------

The solver prints one of the four possible results at termination:

================================  ====================================================================================
Status                            Description
================================  ====================================================================================
:code:`Converged`                 The Algorithm has converged according to the convergence criteria.
:code:`Infeasible`                This happens when the dual objective has exceeded the primal upper bound.
:code:`Terminated`                The solver has reached maximum number of iterations. Users can look at the log to determine if the results are good enough to be used.
:code:`Failed`                    The solver failed during execution. Specific failure reason will be given in logs.
================================  ====================================================================================

We also show the number of iterations, the final primal and dual objective values and the number of active constraints.

Sample log
-----------------

.. code:: text

	------------------------------------------------------------------------
	                          DuaLip v1.0     2021
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
	--input.format avro
	--optimizer.solverType LBFGSB
	--optimizer.dualTolerance 1E-8
	--optimizer.slackTolerance 5E-6
	--optimizer.maxIter 100

	Optimizer: LBFGSB solver
	primalUpperBound: 2.50000000e-05, maxIter: 100, dualTolerance: 1.0E-8 slackTolerance: 5.0E-6

	gradientCall:     0	iter:     0	dual_obj: -4.99999750e+01	cx: -5.00000000e+01	feasibility: 2.217486e+01	λ(Ax-b): 0.000000e+00	γ||x||/2: 2.500000e-05	max_pos_slack: -Infinity	max_zero_slack: 2.217486e+01	abs_slack_sum: 3.146621e+01	time(sec): 5.135
	gradientCall:     1	iter:     1	dual_obj: -4.99999750e+01	cx: -5.00000000e+01	feasibility: 2.217486e+01	λ(Ax-b): 0.000000e+00	γ||x||/2: 2.500000e-05	max_pos_slack: -Infinity	max_zero_slack: 2.217486e+01	abs_slack_sum: 3.146621e+01	time(sec): 1.035
	gradientCall:     2	iter:     1	dual_obj: -1.36522019e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.265221e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 2.079
	gradientCall:     3	iter:     1	dual_obj: -1.36522019e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.265221e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.038
	gradientCall:     4	iter:     2	dual_obj: -1.36522019e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.265221e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.552
	gradientCall:     5	iter:     2	dual_obj: -1.34925665e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.249257e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.945
	gradientCall:     6	iter:     2	dual_obj: -1.34127487e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.241275e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.392
	gradientCall:     7	iter:     2	dual_obj: -1.32930221e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.229303e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.279
	gradientCall:     8	iter:     2	dual_obj: -1.31134322e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.211344e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.580
	gradientCall:     9	iter:     2	dual_obj: -1.28440474e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.184405e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.965
	gradientCall:    10	iter:     2	dual_obj: -1.24399701e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.143998e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.690
	gradientCall:    11	iter:     2	dual_obj: -1.18338541e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -1.083386e+01	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.832
	gradientCall:    12	iter:     2	dual_obj: -1.09246802e+01	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -9.924686e+00	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.559
	gradientCall:    13	iter:     2	dual_obj: -9.56091935e+00	cx: -1.00000000e+00	feasibility: -2.833599e-01	λ(Ax-b): -8.560925e+00	γ||x||/2: 5.400000e-06	max_pos_slack: 2.833599e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.265
	gradientCall:    14	iter:     2	dual_obj: -7.68762043e+00	cx: -2.00000000e+00	feasibility: -2.473638e-01	λ(Ax-b): -5.687626e+00	γ||x||/2: 5.800000e-06	max_pos_slack: 2.473638e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.594
	gradientCall:    15	iter:     2	dual_obj: -7.68762043e+00	cx: -2.00000000e+00	feasibility: -2.473638e-01	λ(Ax-b): -5.687626e+00	γ||x||/2: 5.800000e-06	max_pos_slack: 2.473638e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.231
	gradientCall:    16	iter:     3	dual_obj: -7.68762043e+00	cx: -2.00000000e+00	feasibility: -2.473638e-01	λ(Ax-b): -5.687626e+00	γ||x||/2: 5.800000e-06	max_pos_slack: 2.473638e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.664
	gradientCall:    17	iter:     3	dual_obj: -4.99999750e+01	cx: -5.00000000e+01	feasibility: 2.217486e+01	λ(Ax-b): 0.000000e+00	γ||x||/2: 2.500000e-05	max_pos_slack: -Infinity	max_zero_slack: 2.217486e+01	abs_slack_sum: 3.146621e+01	time(sec): 0.859
	gradientCall:    18	iter:     3	dual_obj: -5.59665447e+00	cx: -7.00000000e+00	feasibility: 1.518062e-01	λ(Ax-b): 1.403338e+00	γ||x||/2: 7.800000e-06	max_pos_slack: 1.518062e-01	max_zero_slack: -Infinity	abs_slack_sum: 2.154136e-01	time(sec): 0.611
	gradientCall:    19	iter:     3	dual_obj: -5.59665447e+00	cx: -7.00000000e+00	feasibility: 1.518062e-01	λ(Ax-b): 1.403338e+00	γ||x||/2: 7.800000e-06	max_pos_slack: 1.518062e-01	max_zero_slack: -Infinity	abs_slack_sum: 2.154136e-01	time(sec): 0.385
	gradientCall:    20	iter:     4	dual_obj: -5.59665447e+00	cx: -7.00000000e+00	feasibility: 1.518062e-01	λ(Ax-b): 1.403338e+00	γ||x||/2: 7.800000e-06	max_pos_slack: 1.518062e-01	max_zero_slack: -Infinity	abs_slack_sum: 2.154136e-01	time(sec): 0.448
	gradientCall:    21	iter:     4	dual_obj: -5.84994265e+00	cx: -4.00000000e+00	feasibility: -1.278211e-01	λ(Ax-b): -1.849949e+00	γ||x||/2: 6.600000e-06	max_pos_slack: 1.278211e-01	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.024
	gradientCall:    22	iter:     4	dual_obj: -5.53061009e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.306171e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.842
	gradientCall:    23	iter:     4	dual_obj: -5.53061009e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.306171e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.021
	gradientCall:    24	iter:     5	dual_obj: -5.53061009e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.306171e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.855
	gradientCall:    25	iter:     5	dual_obj: -5.51271230e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.127193e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.645
	gradientCall:    26	iter:     5	dual_obj: -5.50376340e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.037704e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.730
	gradientCall:    27	iter:     5	dual_obj: -5.50964555e+00	cx: -6.00000000e+00	feasibility: 4.952939e-02	λ(Ax-b): 4.903471e-01	γ||x||/2: 7.400000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 7.028239e-02	time(sec): 0.779
	gradientCall:    28	iter:     5	dual_obj: -5.50037908e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.003861e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.915
	gradientCall:    29	iter:     5	dual_obj: -5.50110192e+00	cx: -6.00000000e+00	feasibility: 4.952939e-02	λ(Ax-b): 4.988907e-01	γ||x||/2: 7.400000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 7.028239e-02	time(sec): 2.274
	gradientCall:    30	iter:     5	dual_obj: -5.50002401e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.000310e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.460
	gradientCall:    31	iter:     5	dual_obj: -5.50012824e+00	cx: -6.00000000e+00	feasibility: 4.952939e-02	λ(Ax-b): 4.998644e-01	γ||x||/2: 7.400000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 7.028239e-02	time(sec): 1.187
	gradientCall:    32	iter:     5	dual_obj: -5.49999490e+00	cx: -6.00000000e+00	feasibility: 4.952939e-02	λ(Ax-b): 4.999977e-01	γ||x||/2: 7.400000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 7.028239e-02	time(sec): 0.341
	gradientCall:    33	iter:     5	dual_obj: -5.49999588e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.000029e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.770
	gradientCall:    34	iter:     5	dual_obj: -5.49999301e+00	cx: -5.82333310e+00	feasibility: 3.202898e-02	λ(Ax-b): 3.233328e-01	γ||x||/2: 7.242060e-06	max_pos_slack: 3.202898e-02	max_zero_slack: -Infinity	abs_slack_sum: 4.544925e-02	time(sec): 0.691
	gradientCall:    35	iter:     5	dual_obj: -5.49999301e+00	cx: -5.82333310e+00	feasibility: 3.202898e-02	λ(Ax-b): 3.233328e-01	γ||x||/2: 7.242060e-06	max_pos_slack: 3.202898e-02	max_zero_slack: -Infinity	abs_slack_sum: 4.544925e-02	time(sec): 0.758
	gradientCall:    36	iter:     6	dual_obj: -5.49999301e+00	cx: -5.82333310e+00	feasibility: 3.202898e-02	λ(Ax-b): 3.233328e-01	γ||x||/2: 7.242060e-06	max_pos_slack: 3.202898e-02	max_zero_slack: -Infinity	abs_slack_sum: 4.544925e-02	time(sec): 0.713
	gradientCall:    37	iter:     6	dual_obj: -5.51201647e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.120235e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.748
	gradientCall:    38	iter:     6	dual_obj: -5.50127583e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.012828e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 1.102
	gradientCall:    39	iter:     6	dual_obj: -5.50012959e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.001366e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.500
	gradientCall:    40	iter:     6	dual_obj: -5.50000727e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.000143e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.686
	gradientCall:    41	iter:     6	dual_obj: -5.49999421e+00	cx: -5.00000000e+00	feasibility: -4.952939e-02	λ(Ax-b): -5.000012e-01	γ||x||/2: 7.000000e-06	max_pos_slack: 4.952939e-02	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.484
	gradientCall:    42	iter:     6	dual_obj: -5.49999295e+00	cx: -5.44549277e+00	feasibility: -5.399420e-03	λ(Ax-b): -5.450721e-02	γ||x||/2: 7.029980e-06	max_pos_slack: 5.399420e-03	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.777
	gradientCall:    43	iter:     6	dual_obj: -5.49999295e+00	cx: -5.44549277e+00	feasibility: -5.399420e-03	λ(Ax-b): -5.450721e-02	γ||x||/2: 7.029980e-06	max_pos_slack: 5.399420e-03	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.735
	gradientCall:    44	iter:     7	dual_obj: -5.49999295e+00	cx: -5.44549277e+00	feasibility: -5.399420e-03	λ(Ax-b): -5.450721e-02	γ||x||/2: 7.029980e-06	max_pos_slack: 5.399420e-03	max_zero_slack: -Infinity	abs_slack_sum: 0.000000e+00	time(sec): 0.925
	gradientCall:    45	iter:     7	dual_obj: -5.49999295e+00	cx: -5.50000000e+00	feasibility: 9.492812e-13	λ(Ax-b): 9.583005e-12	γ||x||/2: 7.050000e-06	max_pos_slack: 9.492812e-13	max_zero_slack: -Infinity	abs_slack_sum: 1.347034e-12	time(sec): 0.934
	gradientCall:    46	iter:     7	dual_obj: -5.49999295e+00	cx: -5.50000000e+00	feasibility: 9.492812e-13	λ(Ax-b): 9.583005e-12	γ||x||/2: 7.050000e-06	max_pos_slack: 9.492812e-13	max_zero_slack: -Infinity	abs_slack_sum: 1.347034e-12	time(sec): 1.155
	Total LBFGS iterations: 7
	Status:Converged
	Total number of iterations: 47
	Primal: -5.44548574018906
	Dual: -5.499992951782625
	Number of Active Constraints: 1
