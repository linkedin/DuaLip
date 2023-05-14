package com.linkedin.dualip.slate

import breeze.linalg.SparseVector
import com.twitter.algebird.Max

/*
 *  This file defines major APIs for slate optimization
 */

/**
 * Object that represents the constructed slate.
 * @param x - value of x variable (in case we allow fractional/probabilistic selections)
 * @param objective - (expected if x!=1) cost of the slate (contribution to the objective function)
 * @param costs - list of items in the slate together with their (expected if x!=1) individual costs (contribution to budget constraints).
 */
case class Slate(x: Double, objective: Double, costs: Seq[(Int, Double)])

/**
 * Generic API of the slate optimizer. It takes the data block, vector of dual variables and
 * outputs one or more slates with their corresponding variable values.
 */
trait SlateComposer {
  /**
   *
   * @param block - data block, usually all parameters related to individual impression
   * @param lambda - vector of dual variables (note the use of java array for performance
   * @return generated slate
   */
  def getSlate(block: MatchingData, lambda: Array[Double]): Seq[Slate]

  /**
   * Get the difference between the max x^Tx and min x^Tx using the defined projection
   * @param block - data block, usually all parameters related to individual impression
   * @return
   */
  def normBound(block: MatchingData): Double = ???

  def sardBound(block: MatchingData, lambda: Array[Double]): (Int, Double, Max[Int]) = ???
}


/**
 * Single slot, first price optimizer. Uses a customm projection.
 * @param gamma          - behaves like a regularizer and controls the smoothness of the objective
 * @param projection     - defines how the primal variable should be projected to respect simple constraints
 */
class SingleSlotComposer(gamma: Double, projection: Projection) extends SlateComposer with Serializable {

  def getProjection: Projection = projection
  /**
   * Normalize the xHat computed (c - a λ) by gamma when we are not using a greedy projection
   * @param xHat
   * @return
   */
  def normalize(xHat: Double): Double = {
    projection match {
      case _: GreedyProjection => xHat
      case _ => xHat / gamma
    }
  }
  /**
   *
   * @param block - data block, usually all parameters related to individual impression
   * @param lambda - vector of dual variables (note the use of java array for performance
   * @return generated slate
   */
  override def getSlate(block: MatchingData, lambda: Array[Double]): Seq[Slate] = {
    val data = block.data.toArray
    val scaledReducedCosts = data.map { case (j, c, a) => normalize(- c - a * lambda(j)) } // compute reduced cost

    projection.project(SparseVector.apply(scaledReducedCosts), block.metadata)
      .activeIterator.map { case (i, x) =>
      val (j, c, a) = data(i)          // we map back the index of the projection to the original item
      Slate(x, c * x, Seq((j, a * x))) // we return expected objective and costs
    }.toSeq
  }

  /**
   * Get max x^Tx - min x^Tx which is used to estimate the next gamma
   * @param block - data block, usually all parameters related to individual impression
   *  @return
   */
  override def normBound(block: MatchingData): Double= {
    val data = block.data.toArray
    projection.maxNorm(data.length) - projection.minNorm(data.length)
  }

  override def sardBound(block: MatchingData, lambda: Array[Double]): (Int, Double, Max[Int]) = {
    val data = block.data.toArray
    val scaledReducedCosts = data.map { case (j, c, a) => normalize(- c - a * lambda(j)) } // compute reduced cost

    val solution = projection.project(SparseVector.apply(scaledReducedCosts), block.metadata)
    val vertex = if (projection.isVertexSolution(solution, block.metadata)) 1 else 0
    (1 - vertex, solution.activeSize, Max(data.length))
  }
}