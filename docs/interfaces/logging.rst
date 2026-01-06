.. _logging:

Logging
=======

Motivation
----------
Reliable experiment tracking is essential for debugging, reproducibility, and collaboration. We recommend using `MLflow <https://mlflow.org>`_ to centralize solver runs, compare configurations, and archive artifacts produced by DuaLip.

How to use MLflow with DuaLip
-----------------------------
DuaLip can be used alongside MLflow to record parameters, per-iteration metrics, and final results. At a high level:

- **Log run parameters** (what you passed to the solver): driver/input/optimizer settings.
- **Log per-iteration metrics** (how optimization is progressing): objective values, feasibility, step size, timing, etc.
- **Log termination summary**: status, iteration counts, primal/dual values, active constraints.
- **Log artifacts**: raw logs and any files produced by your run (e.g., outputs, figures).

What gets logged
----------------
The following metrics are commonly tracked each iteration:

================================  ====================================================================================
Variables                         Description
================================  ====================================================================================
:code:`gradientCall`              Number of gradient evaluations performed so far.
:code:`iter`                      Iteration number (used as MLflow :code:`step`).
:code:`dual_obj`                  Dual objective value (including regularization term).
:code:`cx`                        Primal objective value without regularization, i.e. :math:`c^\top x`.
:code:`feasibility`               :math:`\max_j \{ (Ax-b)_j / (1 + |b_j|)\}`.
:math:`\lambda(Ax-b)`             :math:`\lambda^\top (Ax - b)`.
:math:`\frac{\gamma}{2}||x||^2`   Regularization term.
:code:`max_pos_slack`             :math:`\max_{j: \lambda_j \neq 0} | (Ax-b)_j | / (1 + |b_j|)` (ineq. constraints).
:code:`max_zero_slack`            :math:`\max_{j: \lambda_j = 0} \{ (Ax-b)_j, 0\} / (1 + |b_j|)` (ineq. constraints).
:code:`abs_slack_sum`             Sum of constraint violations.
:code:`time`                      Wall time for the iteration in seconds.
:code:`step`                      Optimizer step size (if applicable).
================================  ====================================================================================

At termination, additionally record:

- **Totals**: final primal and dual objective values and the reqularization penalty.

Quick start
-----------
Install MLflow:

.. code:: bash

	pip install mlflow

(Optional) Configure MLflow: 
If mlflow is not enabled or not configured, MLflow will use the default local configuration.

.. code:: python

	class MLflowConfig:
		"""Configuration for MLflow logging."""

		enabled: bool
		tracking_uri: str = ""
		experiment_name: str = ""
		run_name: str = ""
		log_hyperparameters: bool = True
		log_metrics: bool = True
		synchronous: bool = False


Wrap your solver run with MLflow and log parameters, metrics, and artifacts:

.. code:: python

	import mlflow

	# Example: organize all parameters you pass to DuaLip
	params = {
		"objectiveClass": "matching",
		"gamma": 1e-3,
		"solverType": "AGD",
		"maxIter": 100,
		# Add any other parameters you pass to DuaLip
	}

	with mlflow.start_run(run_name="dualip"):
		# 1) Parameters
		mlflow.log_params(params)

		metrics = {
			"iter": it,
			"dual_obj": -5.5 + it * 1e-3,    # replace with real value
			"cx": -5.0,                      # replace with real value
			"feasibility": 0.02,             # replace with real value
			"max_pos_slack": 0.01,           # replace with real value
			"max_zero_slack": 0.0,           # replace with real value
			"abs_slack_sum": 0.03,           # replace with real value
			"time": 0.7,                     # seconds; replace with real value
			"step": 0.1,                     # replace with real value
		}
			mlflow.log_metrics(metrics, step=it)

		mlflow.log_metrics(
			{
				"total_iters": params["optimizer.maxIter"],
				"primal": -5.44548,
				"dual": -5.49999,
				"num_active_constraints": 1,
			}
		)

Tips
----
- Use :code:`mlflow.log_metrics(..., step=iter)` so MLflow plots metrics over iterations.
