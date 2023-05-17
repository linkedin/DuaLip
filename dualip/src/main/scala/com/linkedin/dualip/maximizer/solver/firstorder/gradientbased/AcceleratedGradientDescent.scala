package com.linkedin.dualip.maximizer.solver.firstorder.gradientbased

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.maximizer.{DualPrimalMaximizer, OptimizerState}
import com.linkedin.dualip.objective.{DualPrimalComputationResult, DualPrimalObjective}
import com.linkedin.dualip.util.IOUtility.{iterationLog, time}
import com.linkedin.dualip.util.SolverUtility.{calculateGroupStepSize, calculateStepSize, expandGroupedStepSize}
import com.linkedin.dualip.util.Status
import com.linkedin.dualip.util.Status.Status

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Implementation of accelerated gradient descent.
  *
  * @param maxIter                   The maximum number of iterations (default is 1000).
  * @param dualTolerance             The dual tolerance limit (default is 1e-6).
  * @param slackTolerance            The slack tolerance limit (default is 0.05).
  * @param designInequality          True if Ax <= b (default), false if Ax = b or have mixed constraints.
  * @param mixedDesignPivotNum       The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many inequality
  *                                  constraints come first (default is 0).
  * @param pivotPositionsForStepSize Pivot positions that mark different groups for which the step-sizes need to be tuned
  *                                  For example, if the total length of the Duals is 10 and we have three groups of
  *                                  sizes 3, 4, and 3 respectively, then pivotPositionsForStepSize must be set at [3, 7].
  */
class AcceleratedGradientDescent(maxIter: Int = 1000,
  dualTolerance: Double = 1e-6,
  slackTolerance: Double = 0.05,
  designInequality: Boolean = true,
  mixedDesignPivotNum: Int = 0,
  pivotPositionsForStepSize: Array[Int] = Array(-1)
) extends Serializable with DualPrimalMaximizer {
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
    *
    * @param f
    * @param initialValue
    * @param verbosity
    * @return
    */
  override def maximize(f: DualPrimalObjective, initialValue: BSV[Double], verbosity: Int = 1):
  (BSV[Double], DualPrimalComputationResult, OptimizerState) = {
    val log = new mutable.StringBuilder()

    val parameters: String = f"AGD solver\nprimalUpperBound: ${f.getPrimalUpperBound}%.8e, " +
      s"maxIter: $maxIter, dualTolerance: $dualTolerance slackTolerance: $slackTolerance" +
      s"\n All norms (||x||) are square norms unless otherwise specified.\n\n"
    print(parameters)
    log ++ parameters

    val iLog = mutable.LinkedHashMap[String, String]()

    // algorithm state variables, updated after each iteration
    var dualObjPrev = 0.0
    var result: DualPrimalComputationResult = null
    var status: Status = Status.Running
    var i = 1
    val useGroupedStepSize = pivotPositionsForStepSize.head != -1
    val dualLength = initialValue.length

    val gradientHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()
    val lambdaHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()

    // (x, y) are used for accelerated update
    var x = initialValue // tracks the dual value
    var y = initialValue // intermediate value to compute acceleration
    // y_new = x + stepSize * grad(f(x))
    // x_new = (1 - beta) * y_new + beta * y
    // dualObj = x^T(A * xhat - b) - c^T * xhat

    while (i <= maxIter && status == Status.Running) {

      iLog.clear()
      iLog += ("iter" -> f"$i%5d")

      // compute function at current dual value
      result = time(f.calculate(x, iLog, verbosity, designInequality, mixedDesignPivotNum), iLog)
      // Check if the dual objective has exceeded the primal upper bound
      if (f.checkInfeasibility(result.lambda, result.dualObjective)) {
        status = Status.Infeasible
      }

      var groupedStepSize = Array[Double]()
      var expandedGroupedStepSize = Array[Double]()
      var stepSize = 0.0
      if (useGroupedStepSize)
        groupedStepSize = calculateGroupStepSize(result.dualGradient.data, y.data, gradientHistory, lambdaHistory,
          pivotPositionsForStepSize)
      else
        stepSize = calculateStepSize(result.dualGradient.data, y.data, gradientHistory, lambdaHistory)

      // log adaptive step size
      if (useGroupedStepSize) {
        iLog += ("step-sizes by groups" -> groupedStepSize.mkString(" "))
        expandedGroupedStepSize = expandGroupedStepSize(pivotPositionsForStepSize, groupedStepSize, dualLength)
      }
      else
        iLog += ("step" -> f"$stepSize%1.2E")

      // check convergence except for first iteration
      val tol = Math.abs(result.dualObjective - dualObjPrev) / Math.abs(dualObjPrev)
      if (i > 1 && tol < dualTolerance && result.slackMetadata.maxSlack < slackTolerance) {
        status = Status.Converged
      } else {
        // if not converged make a gradient step and continue the loop
        // otherwise the loop will break anyway

        var y_new: BSV[Double] = if (useGroupedStepSize) x + (result.dualGradient * BSV(expandedGroupedStepSize))
        else x + (result.dualGradient * stepSize)
        // if we have inequality constraints, then we need to threshold the (dual) variables so that they
        // are non-negative
        y_new = projectOnNNCone(y_new, designInequality, mixedDesignPivotNum)
        x = (y_new * (1.0 - betaSeq(i - 1))) + (y * betaSeq(i - 1))
        y = y_new
        dualObjPrev = result.dualObjective
        i = i + 1
      }

      // write the optimization parameters to a log file
      log ++= iterationLog(iLog)
    }

    status = setStatus(status, i, maxIter)
    (y, result, OptimizerState(i, status, log.toString))
  }
}