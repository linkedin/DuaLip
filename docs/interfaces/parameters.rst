

Parameters
=============

Solver Parameters 
-----------------
The solver settings are displayed in the following table.

=====================================  =============  ==============  ==============================================================================================================
Solevr Arguments                       Is Required    Default Value   Description
=====================================  =============  ==============  ==============================================================================================================
:code:`gamma`                          false          1E-3            Ridge regularization parameter.
:code:`max_iter`                       false          100             Maximum number of iterations.
:code:`initial_step_size`              false          None            Initial step size for the solver.
:code:`max_step_size`                  false          1           	  Upper bound for the solver's step size.
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