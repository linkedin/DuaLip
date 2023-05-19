package com.linkedin.dualip.maximizer

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.objective.{DualPrimalComputationResult, DualPrimalObjective}
import com.linkedin.dualip.util.Status
import com.linkedin.dualip.util.Status.Status

trait DualPrimalMaximizer {

  /**
    * Projects the duals on the non-negative cone for inequality constraints.
    *
    * @param duals               Dual variables.
    * @param designInequality    true if Ax <= b, false if Ax = b or have mixed constraints.
    * @param mixedDesignPivotNum The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many
    *                            inequality constraints come first.
    * @return
    */
  def projectOnNNCone(duals: BSV[Double], designInequality: Boolean, mixedDesignPivotNum: Int): BSV[Double] = {
    for (j <- 0 until duals.length) {
      if (designInequality || j < mixedDesignPivotNum) {
        duals(j) = Math.max(0.0, duals(j))
      }
    }
    duals
  }

  /**
    * Sets convergence status in solvers.
    *
    * @param status  Current status of the solver.
    * @param iter    Number of iterations that the solver has run thus far.
    * @param maxIter Maximum number of iterations that the solver is allowed to run for.
    * @return Updated status.
    */
  def setStatus(status: Status, iter: Int, maxIter: Int): Status.Value = {
    var retStatus = status
    if (status == Status.Running) {
      if (iter >= maxIter)
        retStatus = Status.Terminated
      else
        retStatus = Status.Converged
    }
    retStatus
  }

  /**
    *
    * API of dual-primal gradient optimization methods. The major difference from
    * basic gradient methods (i.e. breeze) is the extra convergence criteria that
    * are specific to primal-dual setup. For example, maximum slack violation.
    *
    * It is possible that this will just be the wrapper for breeze optimizers that
    * handles extra stopping criteria.
    *
    * The API is defined for SparseVector while we most likely will work with dense vectors
    * for dual variable. It should not affect performance because it only affects the
    * meta-optimization algorithm steps, while most of the runtime is spent in distributed
    * objective computation. This objective computation should be very optimized and most
    * likely neither DenseVector, nor SparseVector are going to be used internally.
    * We use Array[Double] in some of objective computation methods.
    *
    * @param f            : Objective function
    * @param initialValue : initial values of the duals
    * @param verbosity    : controls the logging level
    * @return A tuple of (optimizedVariable, objective computation, OptimizerState).
    */
  def maximize(f: DualPrimalObjective, initialValue: BSV[Double], verbosity: Int = 1):
  (BSV[Double], DualPrimalComputationResult, OptimizerState)
}
