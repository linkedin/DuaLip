.. _logging:

Logging
=======

Motivation
----------
Reliable experiment tracking is essential for debugging, reproducibility, and collaboration. We recommend using `MLflow <https://mlflow.org>`_ to centralize solver runs, compare configurations, and archive artifacts produced by DuaLip.

How to use MLflow with DuaLip
-----------------------------
DuaLip can be used alongside MLflow to record parameters, per-iteration metrics, and final results. At a high level:

- **Log run parameters** (what you passed to the solver): By default, the solver logs max_iter, initial_step_size, gamma etc.
- **Log per-iteration metrics** (how optimization is progressing): custom metrics such as feasibility, step size, timing, etc.
- **Log objective result**: the result of the solver including dual objective, primal objective, etc.

See tables below for the variables that are logged by default.

What gets logged
----------------

================================  ====================================================================================
Solver Variables                  Description
================================  ====================================================================================
:code:`max_iter`                  Maximum number of iterations.
:code:`gamma`                     Ridge regularization parameter.
:code:`gamma_decay_type`          Type of gamma decay.
:code:`initial_step_size`         Initial step size.
:code:`max_step_size`             Maximum step size.
================================  ====================================================================================

================================  ====================================================================================
Objective Result                  Description
================================  ====================================================================================
:code:`dual_objective`            Dual objective value.
:code:`primal_objective`          Primal objective value.
:code:`regularization_penalty`    Regularization penalty.
:code:`max_positive_slack`        Maximum positive slack.
:code:`sum_positive_slack`        Sum of positive slack.
================================  ====================================================================================

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

Example configuration to runninng locally:

.. code:: python

	mlflow_config = MLflowConfig(
		enabled=True,
		tracking_uri="http://localhost:5000",
		experiment_name="DuaLip",
		run_name="DuaLip",
	)

Tips
----
- Use :code:`mlflow.log_metrics(..., step=iter)` so MLflow plots metrics over iterations.
