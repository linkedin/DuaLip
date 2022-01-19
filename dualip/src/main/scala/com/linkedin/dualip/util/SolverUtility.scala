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
 
package com.linkedin.dualip.util

import scala.collection.mutable.ListBuffer

/**
  * Util functions shared by solvers
  */
object SolverUtility {

  case class SlackMetadata(slack: Array[Double], maxSlack: Double, maxPosSlack: Double, maxZeroSlack: Double, feasibility: Double)
  /**
    * This function calculates the slack as stopping criteria
    * If lambda_j is 0, then set v_j = max{ (Ax-b)_j, 0} / (1 + abs(b_j)).
    * else, then set v_j = abs((Ax-b)_j) / (1 + abs(b_j)).
    *
    * @param lambda   : vector lambda
    * @param r        : vector ax-b
    * @param b        : vector b
    * @param designInequality : True if Ax <= b, false if Ax = b
    * @return
    */
  def getSlack(lambda: Array[Double], r: Array[Double], b: Array[Double], designInequality: Boolean = true): SlackMetadata = {
    var j = 0
    val res = Array.ofDim[Double](lambda.length)
    var maxPosSlack: Double = Double.NegativeInfinity
    var maxZeroSlack: Double = Double.NegativeInfinity
    var feasibility: Double = Double.NegativeInfinity
    while (j < lambda.length) {
      if (designInequality) {
        if (lambda(j) == 0) {
          res(j) = Math.max(r(j), 0)/(1 + Math.abs(b(j)))
          maxZeroSlack = Math.max(maxZeroSlack, res(j))
        } else {
          res(j) = Math.abs(r(j))/(1 + Math.abs(b(j)))
          maxPosSlack = Math.max(maxPosSlack, res(j))
        }
      }
      else {
        res(j) = Math.abs(r(j))/(1 + Math.abs(b(j)))
        maxPosSlack = Math.max(maxPosSlack, res(j))
      }
      feasibility = Math.max(feasibility, r(j)/(1 + Math.abs(b(j))))
      j = j + 1
    }
    SlackMetadata(res, res.max, maxPosSlack, maxZeroSlack, feasibility)
  }

  /**
    * Calculate the L2 norm of 2 vectors represented as arrays
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
    * Approximate step size calculation based on change in gradient wrt to change in coefficients.
    *
    * λ → gγ (λ) is differentiable and the gradient is Lipschitz continuous with parameter L,
    * i.e., ∥∇gγ(λ)−∇gγ(λ')∥ ≤ L ∥λ−λ'∥ for all λ,λ'.
    * We calculate the max value for L in the history and approximate step size as 1 / L within the specified bounds.
    *
    * @param gradient - The dual gradient
    * @param lambda - The dual variable
    * @param gradientHistory - The gradient history
    * @param lambdaHistory - The dual variable history
    * @param historyLength - The length of the history
    * @param minStepSize - Minimum step size
    * @param maxStepSize - Maximum step size
    * @return
    */
  def calculateStepSize(
    gradient: Array[Double],
    lambda: Array[Double],
    gradientHistory: ListBuffer[Array[Double]],
    lambdaHistory: ListBuffer[Array[Double]],
    historyLength: Int = 15,
    minStepSize: Double = 1e-5,
    maxStepSize: Double = 0.1
  ): Double = {
    if (gradientHistory.length == historyLength) {
      assert(lambdaHistory.length == historyLength, "Gradient and lambda history have diverged.")
      lambdaHistory.remove(0)
      gradientHistory.remove(0)
    }
    lambdaHistory.append(lambda)
    gradientHistory.append(gradient)
    val values: Seq[Double] = (0 until gradientHistory.length - 1)
      .map { i =>
        normOfDifference(gradientHistory(i), gradientHistory(i + 1)) / normOfDifference(lambdaHistory(i), lambdaHistory(i + 1))
      }
    // Allow step size to be computed only if the history is full
    val stepSize: Double = if(values.isEmpty || values.max.isNaN || values.max.isInfinite || values.length < historyLength - 1)
      minStepSize else math.min(1.0 / values.max, maxStepSize)
    stepSize
  }

  /**
   * Uses the DuaLip termination criteria to pick the next value of γ. Refer DuaLip paper for details.
   *
   * @param epsilon - the tolerance to decide convergence set by the adaptive smoothing algorithm
   * @param psi - a bound computed using sard's theorem and the projection information
   * @param g0  - the dual objective corresponding to the "start" of the solver.
   * @param currentSolution - the dual objective corresponding to the "end" of the solver
   * @return
   */
  def calculateGamma(epsilon: Double, psi: Double, g0: Double, currentSolution: Option[Double]): Double = {
    currentSolution match {
      case Some(gLambda) => epsilon / 2 * math.abs(g0 - gLambda) / psi
      case None => epsilon / 2 * math.abs(g0) / psi
    }
  }
}
