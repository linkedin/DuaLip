package com.linkedin.dualip.solver

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.util.OptimizerState

/**
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
  */
trait DualPrimalGradientMaximizer {
  /**
    * API of the solver
    * @param f             Objective function from dual-primal.
    * @param initialValue  Initial lambda value.
    * @param verbosity     Control logging level.
    * @return A tuple of (optimizedVariable, objective computation, OptimizerState).
    */
  def maximize(
    f: DualPrimalDifferentiableObjective,
    initialValue: BSV[Double],
    verbosity: Int = 1
  ): (BSV[Double], DualPrimalDifferentiableComputationResult, OptimizerState)
}