Tips for using DuaLip
==========================
This page contains some things you could try if DuaLip does not work in the way you expected.

**Issue:** The solver converges slowly or does not converge.

Potential things to try:

#. Apply Jacobi Preconditioning: This rescales each constraint constraints by the norm of constraint coefficients. This helps in making the parameters roughly of the same magnitude. (If not, a small set of constraints might dominate the algorithm path.)
#. Try re-running the solver with a larger value of the :code:`gamma` parameter. This helps in smoothing the dual objective function. (For more details, please see the :ref:`solver` section.) You may also use adaptive smoothing by starting with a large gamma and gradually decreasing it over time. This helps in finding a good solution faster while using a smaller gamma.
