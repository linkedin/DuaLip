package com.linkedin.dualip.solver

import breeze.linalg.{SparseVector, min}
import breeze.numerics.abs
import breeze.optimize.{DiffFunction, LBFGS => GenericLBFGS}
import breeze.optimize.FirstOrderMinimizer.State
import breeze.util.Implicits._

import com.linkedin.dualip.util.IOUtility.{iterationLog, time}
import com.linkedin.dualip.util.{OptimizerState, Status}
import com.linkedin.dualip.util.Status.Status

import scala.collection.mutable

/**
  * A custome implementation of LBFGS to solve a maximization problem with non-negativity constraints on the solution
  * @param alpha  a smaller value ensures only non-negative solutions. a large positive value ensures unconstrained optimization
  * @param maxIter is the maximum number of LBFGSB iterations to run
  * @param m       controls the size of the history used
  * @param dualTolerance  change in dual (tolerance) to decide convergence
  * @param slackTolerance change in max slack (tolerance) to decide convergence
  */
class LBFGS(
  alpha: Double,
  maxIter: Int = 100,
  m: Int = 50,
  dualTolerance: Double = 1e-8,
  slackTolerance: Double = 5e-6
) extends Serializable with DualPrimalGradientMaximizer {

  // The number of iterations to hold dual convergence after it has been reached
  // this trick is to ensure the calls from line search don't end up converging the optimizer
  val holdConvergenceForIter: Int = 10

  /**
    * Implementation of the gradient maximizer API for dual-primal solvers
    * @param f - objective function from dual-primal
    * @param initialValue
    * @param verbosity - control logging level
    * @return a tuple of (optimizedVariable, objective computation, number of iterations, log)
    */
  def maximize(f: DualPrimalDifferentiableObjective, initialValue: SparseVector[Double], verbosity: Int)
  : (SparseVector[Double], DualPrimalDifferentiableComputationResult, OptimizerState) = {
    val log = new StringBuilder()
    val lbfgs = new GenericLBFGS[SparseVector[Double]](maxIter = maxIter, m = m)

    // Hold the results of the dual objective from the previous iteration to check convergence
    var lastResult: DualPrimalDifferentiableComputationResult = null
    var lastResultAtIter: Int = 0

    // LBFGS calls this function once before starting the optimization routine to initialize state,
    // we start from 0 and check convergence after iteration 1 to ensure we don't naively converge
    var i: Int = 0
    var trueIter: Int = 0 // this is a true iteration of the algorithm 
    var status: Status = Status.Running

    val parameters: String = f"LBFGS solver\nprimalUpperBound: ${f.getPrimalUpperBound}%.8e, " +
      s"alpha: ${alpha}, maxIter: ${maxIter}, dualTolerance: ${dualTolerance} slackTolerance: ${slackTolerance}\n\n"
    print(parameters)
    log ++ parameters

    val iLog = mutable.LinkedHashMap[String, String]()

    val gradient: DiffFunction[SparseVector[Double]] = new DiffFunction[SparseVector[Double]] {
      // artificially stop the optimizer when the custom convergence criteria has been hit
      val zeros: SparseVector[Double] = SparseVector.zeros(initialValue.length)
      def calculate(x: SparseVector[Double]): (Double, SparseVector[Double]) = {

        iLog.clear()
        iLog += ("gradientCall" -> f"${i}%5d")
        iLog += ("iter" -> f"${trueIter}%5d")

        // compute function at current dual value
        val result: DualPrimalDifferentiableComputationResult = time({
          penalizeNegativity(x, f.calculate(x, iLog, verbosity), iLog)
        }, iLog)

        // write the optimization parameters to a log file
        log ++= iterationLog(iLog)

        // check convergence and print statistics except for first iteration
        if (i > 1) {
          assert(lastResult != null, "last result should have been computed in the previous iteration")
          if (result.slackMetadata.maxSlack < slackTolerance && (i - lastResultAtIter) > holdConvergenceForIter) {
            status = Status.Converged
          }
        }
        // We record the last iteration me made a useful improvement to the objective so that we just check the number
        // of iterations since last useful improvement for convergence. Useful improvement just means the new objective
        // is better than the old by a certain level of tolerance.
        if (lastResult == null || (result.dualObjective - lastResult.dualObjective) / abs(lastResult.dualObjective) > dualTolerance) {
          lastResult = result
          lastResultAtIter = i
        }
        // Check if the dual objective has exceeded the primal upper bound
        if (f.checkInfeasibility(result)) {
          status = Status.Infeasible
        }
        i += 1
        if (status != Status.Running) {
          (-result.dualObjective, zeros)
        } else {
          // LBFGS routine is writen for a minimization problem but the dual is a maximization problem
          (-result.dualObjective, -result.dualGradient)
        }
      }
    }
    val result: State[SparseVector[Double], _, _] = lbfgs.iterations(gradient, initialValue)
      .map { state =>
          trueIter += 1
          state
        }.last
    val iterationMsg: String = s"Total  LBFGS iterations: ${result.iter}\n"
    print(iterationMsg)
    log ++ iterationMsg
    if (status == Status.Running) {
      if (result.iter >= maxIter)
        status = Status.Terminated
      else
        status = Status.Converged
    }
    (result.x, lastResult, OptimizerState(i, status, log.toString))
  }

  /**
    * Use a closed form projection to enforce the non-negativity constraint on lambda
    * @param lambda
    * @param constrainedResult
    * @return
    */
  def penalizeNegativity(lambda: SparseVector[Double], constrainedResult: DualPrimalDifferentiableComputationResult, iLog: mutable.Map[String, String]): DualPrimalDifferentiableComputationResult = {
    val clippedLambda: SparseVector[Double] = min(lambda, 0.0)
    val objective: Double = - (clippedLambda dot clippedLambda) / (2.0 * alpha)
    val gradient: SparseVector[Double] = - clippedLambda / alpha
    iLog += ("-||λ'||/ (2α)" -> f"${objective}%.3f")
    iLog += ("||λ'/α||" -> f"${gradient dot gradient}%.3f")
    constrainedResult.copy(
      dualObjective = constrainedResult.dualObjective + objective,
      dualGradient = constrainedResult.dualGradient + gradient
    )
  }
}