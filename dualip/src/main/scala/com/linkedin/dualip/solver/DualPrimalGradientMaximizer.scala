/*
 * BSD 2-CLAUSE LICENSE
 *
 * Copyright 2021 LinkedIn Corporation
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
 
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