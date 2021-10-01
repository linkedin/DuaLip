package com.linkedin.dualip.projection

import breeze.linalg.{SparseVector => BSV}
import com.linkedin.dualip.projection.VertexType.{Closest, Farthest, VertexType}
import com.linkedin.dualip.util.ProjectionType.BoxCut


/**
 * Boxed simplex projection
 *
 */
class BoxCutProjection(maxIter: Int, inequality: Boolean = false) extends PolytopeProjection(polytope = null, maxIter) with Serializable {

  val unitBoxProjection = new UnitBoxProjection()

  /**
    * Given a vertex (v) that is not inside the polytope find the closest (max dot product) or farthest (min dot product)
    * vertex from v.
    * @param v          - vertex representing primal solution without any polytope constraints
    * @param vertexType - closest or farthest vertex
    * @return
    */
  override def candidateVertex(v: BSV[Double], vertexType: VertexType, metadata: Metadata): (BSV[Double], Double) = {
    val ordering: Ordering[(Int, Double)] = vertexType match {
      case Closest => Ordering.by(pair => pair._2)
      case Farthest => Ordering.by(pair => -pair._2)
    }
    val size: Int = getSize(metadata)
    val pairs: Seq[(Int, Double)] = pickTopN[(Int,Double)](size, v.activeIterator.toSeq)(ordering)
    val vertex: BSV[Double] = new BSV(pairs.map(_._1).toArray.sorted, Array.fill(size){1.0}, v.size)
    val score: Double = pairs.map(_._2).sum
    (vertex, score)
  }

  /**
    * Find the best vertex & check if the best vertex satisfies optimality. If it does, we return
    * it along with its distance from v. If not, we find the second best vertex and return its distance
    * from v
    * @param v - vertex representing primal solution without any polytope constraints
    * @return the top 2 vertices closest to v, along with a check for optimality
    */
  override def checkVertexSolution(v: BSV[Double], metadata: Metadata): VertexSolution = {
    case class Vertex(index: Int, value: Double)
    val size = getSize(metadata)
    // To find the closest vertex to v we sort the values in descending order
    val x: List[Vertex] = v.activeIterator.map(x => Vertex(x._1, x._2)).toList.sortBy(-_.value)
    var currentScore = -size.toDouble
    var currentIndex = Set[Int]()
    for(i <- 0 until math.min(size, x.length)) {
      currentScore += x(i).value
      currentIndex += x(i).index
    }
    // As a sanity check, if you compute the farthest vertex, the distance should be identical
    // val (solution, d) = candidateVertex(v, VertexType.Closest)
    // assert(math.abs(currentDistance - d + alpha) < 0.01)
    val best = currentScore - size
    val bestIndex = currentIndex.toArray
    var secondBest = Double.NegativeInfinity
    var secondBestIndex = Array[Int]()

    for(c <- 1 until math.min(size + 1, x.length - size)) {
      currentScore = currentScore - x(size - c).value + x(size - 1 + c).value + 1
      currentIndex -= x(size - c).index
      currentIndex += x(size - 1 + c).index
      if (currentScore > secondBest) {
        secondBest = currentScore
        secondBestIndex = currentIndex.toArray
      }
    }
    VertexSolution(firstBest = new BSV(bestIndex.sorted, Array.fill(size){1.0}, v.size),
      nextBest = new BSV(secondBestIndex.sorted, Array.fill(size){1.0}, v.size),
      isOptimal = (best > secondBest))
  }

  /**
    * The entry point to project a point onto polytope contraints
    * @param xHat - solution to the primal that does not satisfy polytope constraints
    *  @return projected primal variable
    */
  override def project(xHat: BSV[Double], metadata: Metadata): BSV[Double] = {
    val size = getSize(metadata)
    if (inequality) {
      val intermediate = unitBoxProjection.project(xHat, metadata)
      if (intermediate.data.sum <= size) {
        return intermediate
      }
    }
    equalityConstraint(xHat, metadata)
  }

  override def isVertexSolution(v: BSV[Double], metadata: Metadata): Boolean = {
    val size = getSize(metadata)
    if (inequality) {
      if (v.activeSize <= size && v.data.forall(x => x == 0.0 || x == 1.0)) {
        return true
      }
    } else {
      if (v.activeSize == size && v.data.forall(x => x == 0.0 || x == 1.0)) {
        return true
      }
    }
    false
  }

  /**
   * Size of the box projection is encoded in the key-value metadata.
   * Size is expected to be an integer for this projection
   *
   * @param metadata - Hashmap with auxiliary information per data point (i)
   * @return
   */
  def getSize(metadata: Metadata): Int = {
    metadata.get(BoxCut.toString) match {
      case Some(x) => x.toInt
      case None => throw new IllegalArgumentException("DataBlock metadata should contain boxCut for every record")
    }
  }
}