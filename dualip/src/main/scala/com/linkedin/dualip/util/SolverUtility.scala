package com.linkedin.dualip.util

import scala.collection.mutable.ListBuffer


/**
 * Util functions shared by solvers
 */
object SolverUtility {

  /**
   * This function calculates the slack as stopping criteria
   * If lambda_j is 0, then set v_j = max{ (Ax-b)_j, 0} / (1 + abs(b_j)).
   * else, then set v_j = abs((Ax-b)_j) / (1 + abs(b_j)).
   *
   * @param lambda              : vector lambda
   * @param r                   : vector ax-b
   * @param b                   : vector b
   * @param designInequality    : True if Ax <= b, false if Ax = b or have mixed constraints
   * @param mixedDesignPivotNum : The pivot number if we have mixed A_1x <= b1 and A_2x = b2, i.e. how many inequality constraints come first
   * @return
   */
  def getSlack(lambda: Array[Double], r: Array[Double], b: Array[Double], designInequality: Boolean = true, mixedDesignPivotNum: Int = 0): SlackMetadata = {
    var j = 0
    val res = Array.ofDim[Double](lambda.length)
    var maxPosSlack: Double = Double.NegativeInfinity
    var maxZeroSlack: Double = Double.NegativeInfinity
    var feasibility: Double = Double.NegativeInfinity
    while (j < lambda.length) {
      if (designInequality || j < mixedDesignPivotNum) {
        if (lambda(j) == 0) {
          res(j) = Math.max(r(j), 0) / (1 + Math.abs(b(j)))
          maxZeroSlack = Math.max(maxZeroSlack, res(j))
        } else {
          res(j) = Math.abs(r(j)) / (1 + Math.abs(b(j)))
          maxPosSlack = Math.max(maxPosSlack, res(j))
        }
      }
      else {
        res(j) = Math.abs(r(j)) / (1 + Math.abs(b(j)))
        maxPosSlack = Math.max(maxPosSlack, res(j))
      }
      feasibility = Math.max(feasibility, r(j) / (1 + Math.abs(b(j))))
      j = j + 1
    }
    SlackMetadata(res, res.max, maxPosSlack, maxZeroSlack, feasibility)
  }

  /**
   * Approximate step size calculation based on change in gradient wrt to change in coefficients.
   *
   * λ → gγ (λ) is differentiable and the gradient is Lipschitz continuous with parameter L,
   * i.e., ∥∇gγ(λ)−∇gγ(λ')∥ ≤ L ∥λ−λ'∥ for all λ,λ'.
   * We calculate the max value for L in the history and approximate step size as 1 / L within the specified bounds.
   *
   * @param gradient         - The dual gradient
   * @param lambda           - The dual variable
   * @param gradientHistory  - The gradient history
   * @param lambdaHistory    - The dual variable history
   * @param maxHistoryLength - The length of the history
   * @param minStepSize      - Minimum step size
   * @param maxStepSize      - Maximum step size
   * @return
   */
  def calculateStepSize(
                         gradient: Array[Double],
                         lambda: Array[Double],
                         gradientHistory: ListBuffer[Array[Double]],
                         lambdaHistory: ListBuffer[Array[Double]],
                         maxHistoryLength: Int = 15,
                         minStepSize: Double = 1e-5,
                         maxStepSize: Double = 0.1
                       ): Double = {

    updateDualGradientHistory(gradient, lambda, gradientHistory, lambdaHistory, maxHistoryLength)
    val lipschitzConstants: Seq[Double] = (0 until gradientHistory.length - 1)
      .map { timeIndex =>
        estimateLipschitzConstant(
          gradientHistory(timeIndex),
          gradientHistory(timeIndex + 1),
          lambdaHistory(timeIndex),
          lambdaHistory(timeIndex + 1)
        )
      }
    stepSizeFromLipschitzConstants(lipschitzConstants, maxHistoryLength, minStepSize, maxStepSize)
  }

  /**
   * step-sizes for different groups of contiguous dimensions of the dual vector
   *
   * @param gradient                  - The dual gradient
   * @param lambda                    - The dual variable
   * @param gradientHistory           - The gradient history
   * @param lambdaHistory             - The dual variable history
   * @param pivotPositionsForStepSize - Pivot positions that mark different groups for which the step-sizes need to be tuned
   * @param maxHistoryLength          - The length of the history
   * @param minStepSize               - Minimum step size
   * @param maxStepSize               - Maximum step size
   * @return
   */
  def calculateGroupStepSize(
                              gradient: Array[Double],
                              lambda: Array[Double],
                              gradientHistory: ListBuffer[Array[Double]],
                              lambdaHistory: ListBuffer[Array[Double]],
                              pivotPositionsForStepSize: Array[Int],
                              maxHistoryLength: Int = 15,
                              minStepSize: Double = 1e-5,
                              maxStepSize: Double = 0.1
                            ): Array[Double] = {

    require(!pivotPositionsForStepSize.isEmpty & pivotPositionsForStepSize.head > 0, "pivotPositionsForStepSize " +
      "must be non-empty and have positive entries")

    val dualLength = lambda.length
    val gradientHistoryLength = gradientHistory.length

    // update the historical values of gradients and duals
    updateDualGradientHistory(gradient, lambda, gradientHistory, lambdaHistory, maxHistoryLength)

    // first create an empty Map
    val lipschitzConstantCollection = scala.collection.mutable.Map[Int, List[Double]]()
    (List(0) ++ pivotPositionsForStepSize).foreach(pivotIndex => lipschitzConstantCollection += (pivotIndex -> List[Double]()))

    // collect the Lipschitz constants for different groups and different time indices
    var prevPivotIndex = 0
    (0 until gradientHistoryLength - 1).map { timeIndex =>
      (pivotPositionsForStepSize :+ dualLength).map {
        pivotIndex =>
          if (prevPivotIndex < dualLength) {
            val lipschitzConstant = estimateLipschitzConstant(
              gradientHistory(timeIndex).slice(prevPivotIndex, pivotIndex),
              gradientHistory(timeIndex + 1).slice(prevPivotIndex, pivotIndex),
              lambdaHistory(timeIndex).slice(prevPivotIndex, pivotIndex),
              lambdaHistory(timeIndex + 1).slice(prevPivotIndex, pivotIndex)
            )
            lipschitzConstantCollection += (prevPivotIndex -> (lipschitzConstantCollection(prevPivotIndex) :+ lipschitzConstant))
          }
          prevPivotIndex = pivotIndex
      }
    }

    // create an array of step-sizes, where each group has its own step-size
    prevPivotIndex = 0
    (pivotPositionsForStepSize :+ dualLength).map { pivotIndex =>
      val lipschitzConstants = lipschitzConstantCollection(prevPivotIndex)
      val stepSizeValuesPerGroup = stepSizeFromLipschitzConstants(lipschitzConstants, lipschitzConstants.length, minStepSize, maxStepSize)
      prevPivotIndex = pivotIndex
      stepSizeValuesPerGroup
    }
  }

  /**
   * Update history of the duals and gradients of the duals.
   * If the length of the history exceeds maxHistoryLength, then we remove the first entry of
   * the List.
   *
   * @param gradient
   * @param lambda
   * @param gradientHistory
   * @param lambdaHistory
   * @param maxHistoryLength
   */
  def updateDualGradientHistory(gradient: Array[Double],
                                lambda: Array[Double],
                                gradientHistory: ListBuffer[Array[Double]],
                                lambdaHistory: ListBuffer[Array[Double]],
                                maxHistoryLength: Int): Unit = {
    if (gradientHistory.length == maxHistoryLength) {
      assert(lambdaHistory.length == maxHistoryLength, "Gradient and lambda history have diverged.")
      lambdaHistory.remove(0)
      gradientHistory.remove(0)
    }
    lambdaHistory.append(lambda)
    gradientHistory.append(gradient)
  }


  /**
   * Estimate the Lipschitz constant based on the ratio of the norm differences of the gradient and the norm
   * differences of the duals.
   *
   * @param gradOne - gradient from the first time-index
   * @param gradTwo - gradient from the second time-index
   * @param dualOne - dual from the first time-index
   * @param dualTwo - dual from the second time-index
   * @return
   */
  def estimateLipschitzConstant(gradOne: Array[Double], gradTwo: Array[Double], dualOne: Array[Double],
                                dualTwo: Array[Double]): Double = {
    normOfDifference(gradOne, gradTwo) / normOfDifference(dualOne, dualTwo)
  }

  /**
   * Calculate the L2 norm of 2 vectors represented as arrays
   *
   * @param x the first vector
   * @param y the second vector
   * @return
   */
  private def normOfDifference(x: Array[Double], y: Array[Double]): Double = {
    val sumOfSquares = (x zip y).map {
      case (i, j) => (i - j) * (i - j)
    }.sum
    math.sqrt(sumOfSquares)
  }

  /**
   * Allow step size to be computed only if the history is full
   *
   * @param lipschitzConstants
   * @param maxHistoryLength
   * @param minStepSize
   * @param maxStepSize
   * @return
   */
  def stepSizeFromLipschitzConstants(lipschitzConstants: Seq[Double], maxHistoryLength: Int, minStepSize: Double,
                                     maxStepSize: Double): Double = {
    if (lipschitzConstants.isEmpty || lipschitzConstants.max.isNaN || lipschitzConstants.max.isInfinite ||
      lipschitzConstants.length < maxHistoryLength - 1)
      minStepSize else math.min(1.0 / lipschitzConstants.max, maxStepSize)
  }

  /**
   * Expand grouped step-sizes to the original dual dimensions.
   *
   * @param pivotPositionsForStepSize - Pivot positions that mark different groups for which the step-sizes need to be tuned
   * @param groupedStepSize           - Array of step-sizes per group, there are as many entries in this array as there are
   *                                  number of groups
   * @param dualLength                - Length of the dual vector
   * @return
   */
  def expandGroupedStepSize(pivotPositionsForStepSize: Array[Int], groupedStepSize: Array[Double],
                            dualLength: Int): Array[Double] = {
    // create an array of step-sizes
    var prevPivotIndex = 0
    (pivotPositionsForStepSize :+ dualLength).zipWithIndex.flatMap { case (pivotIndex, index) =>
      val expandedStepSize = Array.fill(pivotIndex - prevPivotIndex)(groupedStepSize(index))
      prevPivotIndex = pivotIndex
      expandedStepSize
    }
  }

  /**
   * Uses the DuaLip termination criteria to pick the next value of γ. Refer DuaLip paper for details.
   *
   * @param epsilon         - the tolerance to decide convergence set by the adaptive smoothing algorithm
   * @param psi             - a bound computed using sard's theorem and the projection information
   * @param g0              - the dual objective corresponding to the "start" of the solver.
   * @param currentSolution - the dual objective corresponding to the "end" of the solver
   * @return
   */
  def calculateGamma(epsilon: Double, psi: Double, g0: Double, currentSolution: Option[Double]): Double = {
    currentSolution match {
      case Some(gLambda) => epsilon / 2 * math.abs(g0 - gLambda) / psi
      case None => epsilon / 2 * math.abs(g0) / psi
    }
  }

  /**
   * maps projection from block metadata
   *
   * @param gamma
   * @param projectionMetadata
   * @return
   */
  def mapProjection(gamma: Double, projectionMetadata: Projection#Metadata): Projection = {
    val projectionName = projectionMetadata.keys.head
    require(projectionName.nonEmpty, "Projection metadata cannot be left blank.")
    ProjectionType.withName(projectionName) match {
      case Simplex =>
        require(gamma > 0, "Gamma should be > 0 for simplex algorithm.")
        new SimplexProjection()
      case SimplexInequality =>
        require(gamma > 0, "Gamma should be > 0 for simplex inequality algorithm.")
        new SimplexProjection(inequality = true)
      case BoxCut =>
        require(gamma > 0, "Gamma should be > 0 for box cut algorithm.")
        new BoxCutProjection(100, inequality = false)
      case BoxCutInequality =>
        require(gamma > 0, "Gamma should be > 0 for box cut inequality algorithm.")
        new BoxCutProjection(100, inequality = true)
      case UnitBox =>
        require(gamma > 0, "Gamma should be > 0 for unit box projection algorithm.")
        new UnitBoxProjection()
      case Greedy =>
        require(gamma == 0, "Gamma should be zero for max element slate optimizer.")
        new GreedyProjection()
      case _ => throw new Exception("Projection " + projectionName + " is not yet supported.")
    }
  }

  case class SlackMetadata(slack: Array[Double], maxSlack: Double, maxPosSlack: Double, maxZeroSlack: Double, feasibility: Double)
}