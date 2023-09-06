Tips for using DuaLip
==========================
This page contains some things you could try if DuaLip does not work in the way you expected.

**Issue:** The solver converges slowly or does not converge.

Potential things to try:

#. Rescale the constraints so that the parameters are roughly of the same magnitude. (If not, a small set of constraints might dominate the algorithm path.)

#. Try re-running the solver with a larger value of the :code:`gamma` parameter.