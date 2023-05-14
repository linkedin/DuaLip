package com.linkedin.dualip.maximizer.solver.firstorder.gradientbased

import breeze.linalg.{SparseVector => BSV}
import breeze.math.InnerProductModule
import breeze.optimize.FirstOrderMinimizer.State
import breeze.optimize.{DiffFunction, FirstOrderException, StochasticDiffFunction, StochasticGradientDescent => GenericSGD}
import com.linkedin.dualip.maximizer.{DualPrimalMaximizer, OptimizerState}
import com.linkedin.dualip.objective.{DualPrimalComputationResult, DualPrimalObjective}
import com.linkedin.dualip.util.IOUtility.{iterationLog, time}
import com.linkedin.dualip.util.Status.Status
import com.linkedin.dualip.util.{SolverUtility, Status}

import scala.collection.mutable
import scala.math.abs

/**
 * A custom implementation of Gradient Descent to solve a maximization problem with non-negativity constraints on the solution
 *
 * @see breeze.optimize.StochasticGradientDescent for the structure of an optimizer
 * @param maxIter        is the maximum number of gradient descent iterations to run
 * @param dualTolerance  change in dual (tolerance) to decide convergence
 * @param slackTolerance change in max slack (tolerance) to decide convergence
 */
class GradientDescent(maxIter: Int = 100,
                      dualTolerance: Double = 1e-8,
                      slackTolerance: Double = 5e-6
                     ) extends Serializable with DualPrimalMaximizer {

  class ProjectedGradientDescent extends GenericSGD[BSV[Double]](
    defaultStepSize = 1E-5, maxIter = maxIter, tolerance = dualTolerance) {
    val GradHist = mutable.ListBuffer[Array[Double]]()
    val XHist = mutable.ListBuffer[Array[Double]]()

    case class Bracket(
                        t: Double, // 1d line search parameter
                        dd: Double, // Directional Derivative at t
                        fval: Double // Function value at t
                      )

    type History = Unit

    def initialHistory(f: StochasticDiffFunction[BSV[Double]], init: BSV[Double]): Unit = ()

    def updateHistory(newX: BSV[Double], newGrad: BSV[Double], newValue: Double, f: StochasticDiffFunction[BSV[Double]], oldState: State): Unit = ()

    def project(x: BSV[Double]): BSV[Double] = {
      x.map(x => if (x < 0) 0.0 else x)
    }


    /**
     * A line search optimizes a function of one variable without analytic gradient information.
     * It's often used approximately (e.g. in backtracking line search), where there is no
     * intrinsic termination criterion, only extrinsic
     *
     * @param f         - actual function to be optimized
     * @param x         - current solution
     * @param direction - direction of gradient
     * @param prod
     * @return
     */
    def functionFromSearchDirection(f: StochasticDiffFunction[BSV[Double]], x: BSV[Double], direction: BSV[Double])
                                   (implicit prod: InnerProductModule[BSV[Double], Double]):
    StochasticDiffFunction[Double] = new StochasticDiffFunction[Double] {

      import prod._

      /** calculates the value at a point */
      override def valueAt(alpha: Double): Double = f.valueAt(project(x + direction * alpha))

      /** calculates the gradient at a point */
      override def gradientAt(alpha: Double): Double = f.gradientAt(project(x + direction * alpha)) dot direction

      /** Calculates both the value and the gradient at a point */
      def calculate(alpha: Double): (Double, Double) = {
        val (ff, grad) = f.calculate(project(x + direction * alpha))
        ff -> (grad dot direction)
      }
    }

    /**
     * Bisection algorithm used for line search to terminate using weak wolfe criterion
     *
     * @param f                 - 1D representation of the function to search over
     * @param init              - initial step size to kick off line search
     * @param maxLineSearchIter - maximum iterations to run line search
     * @return
     */
    def bisectionLineSearch(f: StochasticDiffFunction[Double], init: Double = 1.0, maxLineSearchIter: Int): Double = {
      def phi(t: Double): Bracket = {
        val (pval, pdd) = f.calculate(t)
        Bracket(t = t, dd = pdd, fval = pval)
      }

      val c1 = 1e-4
      val c2 = 0.99

      var t = init // Search's current multiple of pk
      val low = phi(0.0)
      val fval = low.fval
      val dd = low.dd

      if (dd > 0) {
        throw new FirstOrderException("Line search invoked with non-descent direction: " + dd)
      }

      var alpha: Double = 0.0
      var beta: Double = Double.PositiveInfinity

      var weakWolfeConditions = true
      var iter = 0
      while (weakWolfeConditions && iter < maxLineSearchIter) {
        val c = phi(t)
        if ((c.fval > fval + c1 * t * dd) || (c.dd < c2 * dd)) {
          if (c.fval > fval + c1 * t * dd) {
            beta = t
            t = (alpha + beta) / 2.0
          } else {
            alpha = t
            if (java.lang.Double.isInfinite(beta) || java.lang.Double.isNaN(beta)) {
              t = 2.0 * alpha
            } else {
              t = (alpha + beta) / 2.0
            }
          }
        } else {
          weakWolfeConditions = false
        }
        iter += 1
      }
      t
    }

    override def determineStepSize(state: State, f: StochasticDiffFunction[BSV[Double]], dir: BSV[Double]): Double = {
      val x = state.x
      val grad = state.grad
      val ff = functionFromSearchDirection(f, x, dir)

      val init = SolverUtility.calculateStepSize(grad.data, x.data, GradHist, XHist)
      bisectionLineSearch(ff, init, 20)
    }

    override protected def takeStep(state: State, dir: BSV[Double], stepSize: Double): BSV[Double] = {
      project(state.x + dir * stepSize)
    }

    override protected def chooseDescentDirection(state: State, fn: StochasticDiffFunction[BSV[Double]]): BSV[Double] = {
      state.grad * -1.0
    }
  }

  // The number of iterations to hold dual convergence after it has been reached
  // this trick is to ensure the calls from line search don't end up converging the optimizer
  val holdConvergenceForIter: Int = 20

  /**
   * Implementation of the gradient maximizer API for dual-primal solvers
   *
   * @param f         - objective function from dual-primal
   * @param initialValue
   * @param verbosity - control the logging level
   * @return a tuple of (optimizedVariable, objective computation, number of iterations, log)
   */
  override def maximize(f: DualPrimalObjective, initialValue: BSV[Double], verbosity: Int = 1):
  (BSV[Double], DualPrimalComputationResult, OptimizerState) = {
    val log = new mutable.StringBuilder()
    val sgd = new ProjectedGradientDescent()

    // Hold the results of the dual objective from the previous iteration to check convergence
    var lastResult: DualPrimalComputationResult = null
    var lastResultAtIter: Int = 0

    // ProjectedGradientDescent calls this function once before starting the optimization routine to initialize state,
    // we start from 0 and check convergence after iteration 1 to ensure we don't naively converge
    var i: Int = 0
    var status: Status = Status.Running

    val parameters: String = f"GD solver\nprimalUpperBound: ${f.getPrimalUpperBound}%.8e, maxIter: $maxIter, " +
      s"dualTolerance: $dualTolerance slackTolerance: $slackTolerance\n\n"
    print(parameters)
    log ++ parameters

    // To preserve the order of insertion
    val iLog = mutable.LinkedHashMap[String, String]()

    val gradient: DiffFunction[BSV[Double]] = new DiffFunction[BSV[Double]] {
      // artificially stop the optimizer when the custom convergence criteria has been hit
      val zeros: BSV[Double] = BSV.zeros(initialValue.length)

      def calculate(x: BSV[Double]): (Double, BSV[Double]) = {

        iLog.clear()
        iLog += ("iter" -> f"$i")

        // compute function at current dual value
        val result: DualPrimalComputationResult = time({
          f.calculate(x, iLog, verbosity)
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
          // Check if the dual objective has exceeded the primal upper bound
          if (f.checkInfeasibility(result.lambda, result.dualObjective)) {
            status = Status.Infeasible
          }
        }
        i += 1
        if (status != Status.Running) {
          (-result.dualObjective, zeros)
        } else {
          // optimizer routines are writen for a minimization problem but the dual is a maximization problem
          (-result.dualObjective, -result.dualGradient)
        }
      }
    }
    val result: State[BSV[Double], _, _] = sgd.minimizeAndReturnState(gradient, initialValue)
    val iterationMsg: String = s"Total iterations: ${result.iter}\n"
    print(iterationMsg)
    log ++ iterationMsg

    status = setStatus(status, result.iter, maxIter)
    (result.x, lastResult, OptimizerState(result.iter, status, log.toString))
  }
}
