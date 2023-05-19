package com.linkedin.dualip.slate

import breeze.linalg.SparseVector
import com.linkedin.dualip.data.MultipleMatchingData
import com.linkedin.dualip.projection.{GreedyProjection, Projection}
import com.twitter.algebird.Max

/**
  * class for constrained matching slate optimizer
  *
  * @param gamma      : weight for the squared loss term
  * @param projection : type of projection
  */
class MultipleMatchingSlateComposer(gamma: Double, projection: Projection) extends Serializable {

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
    * @param matchingConstraintsPerIndex
    * @return
    */
  def computeReducedCosts(data: Array[(Int, Double, Seq[(Int, Double)])], duals: Array[Double],
    matchingConstraintsPerIndex: Int):
  Array[Double] = {
    data.map { case (entityIndex, c, valuesA) =>
      normalize(-c - valuesA.map {
        case (matchingConstraintIndex, valueA) => duals(matchingConstraintsPerIndex * entityIndex +
          matchingConstraintIndex) * valueA
      }.sum)
    }
  }

  /**
    * computes primal values and creates the slates
    *
    * @param block
    * @param duals
    * @param matchingConstraintsPerIndex
    * @return
    */
  def getSlate(block: MultipleMatchingData, duals: Array[Double], matchingConstraintsPerIndex: Int): Seq[Slate] = {
    val data = block.data.toArray
    val scaledReducedCosts = computeReducedCosts(data, duals, matchingConstraintsPerIndex)

    projection.project(SparseVector.apply(scaledReducedCosts), block.metadata)
      .activeIterator.map { case (blockIndex, x) =>
      val (entityIndex, c, valuesA) = data(blockIndex)
      Slate(x, c * x, valuesA.map { case (matchingConstraintIndex, valueA) =>
        (matchingConstraintsPerIndex * entityIndex + matchingConstraintIndex, valueA * x)
      })
    }.toSeq
  }

  /**
    * Get max x^Tx - min x^Tx which is used to estimate the next gamma
    *
    * @param block
    * @return
    */
  def normBound(block: MultipleMatchingData): Double = {
    val data = block.data.toArray
    projection.maxNorm(data.length) - projection.minNorm(data.length)
  }

  /**
    *
    * @param block
    * @param duals
    * @return
    */
  def sardBound(block: MultipleMatchingData, duals: Array[Double], matchingConstraintsPerIndex: Int):
  (Int, Double, Max[Int]) = {
    val data = block.data.toArray
    val scaledReducedCosts = computeReducedCosts(data, duals, matchingConstraintsPerIndex)
    val solution = projection.project(SparseVector.apply(scaledReducedCosts), block.metadata)
    val vertex = if (projection.isVertexSolution(solution, block.metadata)) 1 else 0
    (1 - vertex, solution.activeSize, Max(data.length))
  }
}