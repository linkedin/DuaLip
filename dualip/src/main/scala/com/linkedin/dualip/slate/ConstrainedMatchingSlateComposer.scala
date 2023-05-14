package com.linkedin.dualip.slate

import breeze.linalg.SparseVector
import com.twitter.algebird.Max

/**
 * class for constrained matching slate optimizer
 *
 * @param gamma      : weight for the squared loss term
 * @param projection : type of projection
 */
class ConstrainedMatchingSlateComposer(gamma: Double, projection: Projection) extends Serializable {

  def getProjection: Projection = projection

  /**
   * normalize the xHat by gamma when except for the case of greedy projection
   *
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
   * computes the primal variables
   *
   * @param data
   * @param duals
   * @return
   */
  def computeReducedCosts(data: Array[(Int, Double, Double, Seq[(Int, Double)])], duals: ConstrainedMatchingDuals):
  Array[Double] = {
    data.map { case (localConstraintIndex, c, localA, globalAs) =>
      normalize(-c - localA * duals.lambdaLocal(localConstraintIndex) - globalAs.map {
        case (globalConstraintIndex, globalA) => duals.lambdaGlobal(globalConstraintIndex) * globalA
      }.sum)
    }
  }

  /**
   * computes primal values and creates the slates
   *
   * @param block
   * @param duals
   * @return
   */
  def getSlate(block: ConstrainedMatchingData, duals: ConstrainedMatchingDuals): Seq[Slate] = {
    val data = block.data.toArray
    val scaledReducedCosts = computeReducedCosts(data, duals)
    val globalLambdaLength = duals.lambdaGlobal.length

    projection.project(SparseVector.apply(scaledReducedCosts), block.metadata)
      .activeIterator.map { case (blockIndex, x) =>
      val (localConstraintIndex, c, localA, globalAs) = data(blockIndex)
      Slate(x, c * x, (localConstraintIndex + globalLambdaLength, localA * x) +:
        globalAs.map { case (globalConstraintIndex, g) => (globalConstraintIndex, g * x)
        })
    }.toSeq
  }

  /**
   * Get max x^Tx - min x^Tx which is used to estimate the next gamma
   *
   * @param block
   * @return
   */
  def normBound(block: ConstrainedMatchingData): Double = {
    val data = block.data.toArray
    projection.maxNorm(data.length) - projection.minNorm(data.length)
  }

  /**
   *
   * @param block
   * @param duals
   * @return
   */
  def sardBound(block: ConstrainedMatchingData, duals: ConstrainedMatchingDuals): (Int, Double, Max[Int]) = {
    val data = block.data.toArray
    val scaledReducedCosts = computeReducedCosts(data, duals)
    val solution = projection.project(SparseVector.apply(scaledReducedCosts), block.metadata)
    val vertex = if (projection.isVertexSolution(solution, block.metadata)) 1 else 0
    (1 - vertex, solution.activeSize, Max(data.length))
  }
}