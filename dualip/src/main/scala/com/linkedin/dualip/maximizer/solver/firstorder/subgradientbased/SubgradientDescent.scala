package com.linkedin.dualip.maximizer.solver.firstorder.subgradientbased

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.maximizer.{DualPrimalMaximizer, OptimizerState}
import com.linkedin.dualip.objective.{DualPrimalComputationResult, DualPrimalObjective}
import com.linkedin.dualip.util.IOUtility.{iterationLog, time}
import com.linkedin.dualip.util.SolverUtility.calculateStepSize
import com.linkedin.dualip.util.Status
import com.linkedin.dualip.util.Status.Status

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * Implementation of subgradient descent.
 *
 * @param maxIter             The maximum number of iterations (default is 1000).
 * @param dualTolerance       The dual tolerance limit (default is 1e-6).
 * @param slackTolerance      The slack tolerance limit (default is 0.05).
 * @param designInequality    True if Ax <= b (default), false if Ax = b or have mixed constraints.
 * @param mixedDesignPivotNum The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many inequality
 *                            constraints come first (default is 0).
 */
class SubgradientDescent(maxIter: Int = 1000,
                         dualTolerance: Double = 1e-6,
                         slackTolerance: Double = 0.05,
                         designInequality: Boolean = true,
                         mixedDesignPivotNum: Int = 0)
  extends Serializable with DualPrimalMaximizer {


  // The number of iterations to hold dual convergence after it has been reached
  // this trick is to ensure the calls from line search don't end up converging the optimizer
  val holdConvergenceForIter: Int = 20

  /**
   * Implementation of the gradient maximizer API for dual-primal solvers
   *
   * @param f            Objective function from dual-primal.
   * @param initialValue Initial value for the dual variable.
   * @param verbosity    Control logging level.
   * @return A tuple of (optimizedVariable, objective computation, OptimizerState).
   */
  override def maximize(f: DualPrimalObjective, initialValue: BSV[Double], verbosity: Int = 1):
  (BSV[Double], DualPrimalComputationResult, OptimizerState) = {
    val log = new mutable.StringBuilder()

    val parameters: String = f"Sub-gradient descent solver\nprimalUpperBound: ${f.getPrimalUpperBound}%.8e, " +
      s"maxIter: $maxIter, dualTolerance: $dualTolerance slackTolerance: $slackTolerance" +
      s"\n All norms (||x||) are square norms unless otherwise specified.\n\n"
    print(parameters)
    log ++ parameters

    val iLog = mutable.LinkedHashMap[String, String]()

    // algorithm state variables, updated after each iteration
    var dualObjPrev = 0.0
    var result: DualPrimalComputationResult = null
    var status: Status = Status.Running
    var iter: Int = 1

    var x: BSV[Double] = initialValue
    val gradientHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()
    val lambdaHistory: ListBuffer[Array[Double]] = mutable.ListBuffer[Array[Double]]()

    while (iter <= maxIter && status == Status.Running) {

      iLog.clear()
      iLog += ("iter" -> f"$iter%5d")

      // compute function at current dual value
      result = time(f.calculate(x, iLog, verbosity, designInequality, mixedDesignPivotNum), iLog)
      // Check if the dual objective has exceeded the primal upper bound
      if (f.checkInfeasibility(result.lambda, result.dualObjective)) {
        status = Status.Infeasible
      }

      // calculate step-size
      val stepSize = calculateStepSize(result.dualGradient.data, result.lambda.data, gradientHistory, lambdaHistory)

      // log adaptive step size
      iLog += ("step" -> f"$stepSize%1.2E")

      // check convergence except for first iteration
      val tol = Math.abs(result.dualObjective - dualObjPrev) / Math.abs(dualObjPrev)
      if (iter > 1 && tol < dualTolerance && result.slackMetadata.maxSlack < slackTolerance) {
        status = Status.Converged
      } else {
        // if not converged make a gradient step and continue the loop
        // otherwise the loop will break anyway
        x += (result.dualGradient * stepSize)
        // if we have inequality constraints, then we need to threshold the (dual) variables so that they
        // are non-negative
        x = projectOnNNCone(x, designInequality, mixedDesignPivotNum)
        dualObjPrev = result.dualObjective
        iter += 1
      }

      // write the optimization parameters to a log file
      log ++= iterationLog(iLog)
    }
    status = setStatus(status, iter, maxIter)
    (x, result, OptimizerState(iter, status, log.toString))
  }
}
