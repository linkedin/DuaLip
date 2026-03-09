

Parameters
=============
The main interface for running the solver is :code:`run_solver()` function. It takes the following arguments:

- **input_args**: Input arguments for the solver.
- **solver_args**: Solver arguments for the solver.
- **compute_args**: Compute arguments for the solver.
- **objective_args**: Objective arguments for the solver.  
- **mlflow_config**: MLflow configuration for the solver.

Input Arguments
---------------
The input arguments are displayed in the following table.

=====================================  =============  ==============  ==============================================================================================================
Input Arguments                        Is Required    Default Value   Description
=====================================  =============  ==============  ==============================================================================================================
:code:`A`                              true           N/A             Tensor A.
:code:`b`                              true           N/A             Tensor b.
:code:`C`                              true           N/A             Tensor C.
:code:`projection_maps`                true           N/A             Projection maps.
=====================================  =============  ==============  ==============================================================================================================

Solver Parameters 
-----------------
The solver settings are displayed in the following table.

=====================================  =============  ==============  ==============================================================================================================
Solevr Arguments                       Is Required    Default Value   Description
=====================================  =============  ==============  ==============================================================================================================
:code:`gamma`                          false          1E-3            Ridge regularization parameter.
:code:`max_iter`                       false          10000           Maximum number of iterations.
:code:`initial_step_size`              false          1e-5            Initial step size for the solver.
:code:`max_step_size`                  false          0.1             Upper bound for the solver's step size.
:code:`save_primal`                    false          false           Flag to save primal variable values at the solution.
:code:`initial_dual_path`              false          None            Filepath to initialize dual variables for algorithm restarts (optional).
:code:`gamma_decay_type`               false          None            Type of gamma decay. We currently support "none" and "step".
:code:`gamma_decay_params`             false          None            Parameters for gamma decay. For "step" type gamme decay, "decay_steps" and "decay_rate" are the tunable parameters.
:code:`decay_steps`                    false          None            Number of steps before decaying gamma. This is only applicable when "step" type gamma decay is used.
:code:`decay_rate`                     false          None            Rate of gamma decay. This is only applicable when "step" type gamma decay is used.
=====================================  =============  ==============  ==============================================================================================================

Compute Parameters 
-----------------
The compute settings are displayed in the following table.

=====================================  =============  ==============  ==============================================================================================================
Compute Arguments                      Is Required    Default Value   Description
=====================================  =============  ==============  ==============================================================================================================
:code:`host_device`                    true           N/A             Host device to compute the gradients. Choose from 'cuda:0' or 'cpu' with cuda:0 used when cuda is available.
:code:`compute_device_num`             false          1               Number of compute devices to use for gradient computation.
=====================================  =============  ==============  ==============================================================================================================

The most efficient parameters are dependent on the problem scale and structure and there is no one-size-fits-all 
configuration. The default parameters usually can achieve acceptable performance, but users can definitely try tuning them if needed.  


Objective Arguments
-------------------
The objective arguments are displayed in the following table.

=====================================  =============  ==============  ==============================================================================================================
Objective Arguments                    Is Required    Default Value   Description
=====================================  =============  ==============  ==============================================================================================================
:code:`objective_type`                 true           N/A             Type of objective function.
:code:`objective_params`               false          None            Parameters for the objective function.
=====================================  =============  ==============  ==============================================================================================================

MLflow Configuration
--------------------
The MLflow configuration is displayed in the following table.

=====================================  =============  ==============  ==============================================================================================================
MLflow Configuration                    Is Required    Default Value   Description
=====================================  =============  ==============  ==============================================================================================================
:code:`enabled`                          true           N/A             Whether to enable MLflow logging.
:code:`tracking_uri`                     false          None            MLflow tracking URI.
:code:`experiment_name`                  false          None            MLflow experiment name.
:code:`run_name`                         false          None            MLflow run name.
=====================================  =============  ==============  ==============================================================================================================

The function :code:`run_solver()` returns a :code:`SolverResult` object.

SolverResult
------------
The solver result is displayed in the following table.

=====================================  ==============  ==============================================================================================================
Solver Result                          Default Value   Description
=====================================  ==============  ==============================================================================================================
:code:`dual_val`                       N/A             Dual variable value.
:code:`dual_objective`                 N/A             Dual objective value.
:code:`objective_result`               N/A             The result of one gradient call.
:code:`dual_objective_log`             N/A             Log of dual objective values.
:code:`step_size_log`                  N/A             Log of step sizes.
=====================================  ==============  ==============================================================================================================

dual_objective_result is an object of the class :code:`ObjectiveResult` and it contains the following attributes:

- :code:`dual_gradient`: Gradient of the dual objective function.
- :code:`dual_objective`: Dual objective value.
- :code:`reg_penalty`: Regularization penalty.
- :code:`primal_objective`: Primal objective value.
- :code:`primal_var`: Primal variable value.
- :code:`dual_val_times_grad`: Dual variable times gradient.
- :code:`max_pos_slack`: Maximum positive slack.
- :code:`sum_pos_slack`: Sum of positive slack.


