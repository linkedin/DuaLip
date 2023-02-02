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
import com.linkedin.dualip.util.IOUtility.{iterationLog, time}
import com.linkedin.dualip.util.Status.Status
import com.linkedin.dualip.util.{OptimizerState, SolverUtility, Status}
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Implementation of accelerated gradient descent.
  * @param maxIter              The maximum number of iterations (default is 1000).
  * @param dualTolerance        The dual tolerance limit (default is 1e-6).
  * @param slackTolerance       The slack tolerance limit (default is 0.05).
  * @param designInequality     True if Ax <= b (default), false if Ax = b or have mixed constraints.
  * @param mixedDesignPivotNum  The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many inequality
  *                             constraints come first (default is 0).
  */
class AcceleratedGradientDescent(
  maxIter: Int = 1000,
  dualTolerance: Double = 1e-6,
  slackTolerance: Double = 0.05,
  designInequality: Boolean = true,
  mixedDesignPivotNum: Int = 0
) extends Serializable with DualPrimalGradientMaximizer {
  // Initialize betaSeq sequence for acceleration
  val betaSeq: Array[Double] = {
    val tseq = Array.ofDim[Double](maxIter + 2)
    val betaSeq = Array.ofDim[Double](maxIter)

    for (i <- 1 until (maxIter + 2)) {
      tseq(i) = (1 + Math.sqrt(1 + 4 * Math.pow(tseq(i - 1), 2))) / 2
    }
    for (i <- 0 until maxIter) {
      betaSeq(i) = (1 - tseq(i + 1)) / tseq(i + 2)
    }
    betaSeq
  }

  /**
    * Implementation of the gradient maximizer API for dual-primal solvers
    * @param f             Objective function from dual-primal.
    * @param initialValue  Initial value for the dual variable.
    * @param verbosity     Control logging level.
    * @return A tuple of (optimizedVariable, objective computation, OptimizerState).
    */
  override def maximize(f: DualPrimalDifferentiableObjective, initialValue: BSV[Double], verbosity: Int)
  : (BSV[Double], DualPrimalDifferentiableComputationResult, OptimizerState) = {
    val log = new StringBuilder()

    val parameters: String = f"AGD solver\nprimalUpperBound: ${f.getPrimalUpperBound}%.8e, " +
      s"maxIter: $maxIter, dualTolerance: $dualTolerance slackTolerance: $slackTolerance" +
      s"\n All norms (||x||) are square norms unless otherwise specified.\n\n"
    print(parameters)
    log ++ parameters

    val iLog = mutable.LinkedHashMap[String, String]()

    // algorithm state variables, updated after each iteration
    var dualObjPrev = 0.0
    var result: DualPrimalDifferentiableComputationResult = null
    var status: Status = Status.Running
    var i = 1

    val gradientHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]] ()
    val lambdaHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]] ()

    // (x, y) are used for accelerated update
    var x = initialValue // tracks the dual value
    var y = initialValue // intermediate value to compute acceleration
    // y_new = x + stepSize * grad(f(x))
    // x_new = (1 - beta) * y_new + beta * y
    // dualObj = x^T(A * xhat - b) - c^T * xhat

    while (i <= maxIter && status == Status.Running) {

      iLog.clear()
      iLog += ("iter" -> f"${i}%5d")

      // compute function at current dual value
      result = time(f.calculate(x, iLog, verbosity, designInequality, mixedDesignPivotNum), iLog)
      // Check if the dual objective has exceeded the primal upper bound
      if (f.checkInfeasibility(result)) {
        status = Status.Infeasible
      }

      val stepSize = SolverUtility.calculateStepSize(
        result.dualGradient.data,
        y.data,
        gradientHistory,
        lambdaHistory)

      // log adaptive step size
      iLog += ("step" -> f"$stepSize%1.2E")

      // check convergence except for first iteration
      val tol = Math.abs(result.dualObjective - dualObjPrev) / Math.abs(dualObjPrev)
      if (i > 1 && tol < dualTolerance && result.slackMetadata.maxSlack < slackTolerance) {
        status = Status.Converged
      } else {
        // if not converged make a gradient step and continue the loop
        // otherwise the loop will break anyway
        val y_new: BSV[Double] = x + (result.dualGradient * stepSize)
        // if we have inequality constraints, then we need to threshold the (dual) variables so that they
        // are non-negative
        for (j <- 0 until x.length) {
          if (designInequality || j < mixedDesignPivotNum) {
            y_new(j) = if (y_new(j) < 0) 0.0 else y_new(j)
          }
        }

        x = (y_new * (1.0 - betaSeq(i - 1))) + (y * betaSeq(i - 1))
        y = y_new
        dualObjPrev = result.dualObjective
        i = i + 1
      }

      // write the optimization parameters to a log file
      log ++= iterationLog(iLog)
    }
    if (status == Status.Running) {
      if (i >= maxIter)
        status = Status.Terminated
      else
        status = Status.Converged
    }

    (y, result, OptimizerState(i, status, log.toString))
  }
}