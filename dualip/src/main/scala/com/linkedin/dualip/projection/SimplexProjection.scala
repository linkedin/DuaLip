package com.linkedin.dualip.projection

import breeze.linalg.{SparseVector => BSV}

/**
 * Simplex projection, \sum_j x_j = 1 or \sum_j x_j <= 1 along with 0 <= x_j <= 1
 *
 * @param checkVertexSolution: check's for vertex solutions before projecting. is faster in practice
 * @param inequality: makes the constraint \sum_j x_j <= 1 when true
 *
 */
class SimplexProjection(checkVertexSolution: Boolean = true, inequality: Boolean = false) extends Projection with Serializable {

  val unitBoxProjection = new UnitBoxProjection()

  /** Project a vector onto the L1 Ball.
    *
    * Implement a modified version of algorithm in Figure 1 in [1].
    * Include an additional pair (-1, 0) to the sparse set to represent zeros
    * 1. Sort v into u in descending order
    * 2. rho = max{j in [n], uj - 1/j*(sum_{1 to j} - 1) > 0}
    * 3. theta = 1/p(sum_{1 to rho}(ui) - 1)
    * 4. Output w where wi = max{vi-theta, 0}
    *
    * [1]. Duchi, John, et al. "Efficient projections onto the L1-ball for learning in high dimensions."
    * Proceedings of the 25th international conference on Machine learning. ACM, 2008.
    *
    * @param v      Input vector v
    */
  def simplex(v: BSV[Double]): BSV[Double] = {
    // add (-1, 0) pair to represent zeros
    val idx = v.index :+ -1
    val values = v.data :+ 0.0

    // Step 1
    val sorted = idx.zip(values).sortBy(_._2)(Ordering[Double].reverse)

    // Step 2
    var i = 0
    var rho = 0
    var prevRho = 0
    var runningSum = 0.0
    var runningSumToRho = 0.0
    var positive = true
    while (i < sorted.length && positive) {
      prevRho = rho
      runningSum = runningSum + sorted(i)._2
      if (sorted(i)._1 == -1)
        rho = rho + v.size - v.data.length
      else
        rho = rho + 1
      val t = sorted(i)._2 - 1.0 / rho * (runningSum - 1)
      if (t <= 0) {
        rho = prevRho
        positive = false
      }
      if (t > 0) {
        runningSumToRho = runningSum
      }
      i += 1
    }

    // Step 3
    val theta = (runningSumToRho - 1) / rho

    // Step 4
    val projected = v.data.map(x => if (x - theta > 0) x - theta else 0)

    val result = new BSV(v.index, projected, v.size)
    // compact() will have all explicit zeros removed.
    result.compact()
    result
  }

  def modifiedDuchi(v: BSV[Double]): BSV[Double] = {
    // add (-1, 0) pair to represent zeros
    val idxs = v.index :+ -1
    val values = v.data :+ 0.0

    // Step 1
    val queue = collection.mutable.PriorityQueue[(Int, Double)](idxs.zip(values): _*)(Ordering.by(_._2))

    // Step 2
    var i = 0
    var rho = 0
    var prevRho = 0
    var runningSum = 0.0
    var runningSumToRho = 0.0
    var positive = true
    while (i < idxs.length && positive) {
      prevRho = rho
      val (idx, value) = queue.dequeue()
      runningSum = runningSum + value
      if (idx == -1)
        rho = rho + v.size - v.data.length
      else
        rho = rho + 1
      val t = value - 1.0 / rho * (runningSum - 1)
      if (t <= 0) {
        rho = prevRho
        positive = false
      }
      if (t > 0) {
        runningSumToRho = runningSum
      }
      i += 1
    }

    // Step 3
    val theta = (runningSumToRho - 1) / rho

    // Step 4
    val projected = v.data.map(x => if (x - theta > 0) x - theta else 0)

    val result = new BSV(v.index, projected, v.size)
    // compact() will have all explicit zeros removed.
    result.compact()
    result
  }

  /**
   * Appendix of DuaLip paper has the proof. Check if the closest vertex (f) is optimal. This will hold iff
   * there is no feasible descent direction at the f,  i.e. the  negative  gradient  at f makes
   * an obtuse angle with all feasible directions.
   *
   * @param v vector to be projected
   * @return
   */
  def checkVertexSolution(v: BSV[Double]): (BSV[Double], Boolean) = {
    var first = Double.MinValue
    var second = Double.MinValue
    var idx = -1
    var i = 0
    while (i < v.data.length) {
      if (v.data(i) > first) {
        second = first
        first = v.data(i)
        idx = i
      } else if (v.data(i) > second) {
        second = v.data(i)
      }
      i = i + 1
    }
    val isOptimal = second < first - 1
    val vertex = new BSV(Array(idx), Array(1.0), v.size)
    (vertex, isOptimal)
  }

  /**
   * Implement the projection for \sum_j x_j = 1
   * @param v vector to be projected
   * @return
   */
  def equalityConstraint(v: BSV[Double]): BSV[Double] = {
    if (checkVertexSolution) {
      val (vertex, isOptimal) = checkVertexSolution(v)
      if (isOptimal) {
        return vertex
      }
      modifiedDuchi(v)
    } else {
      simplex(v)
    }
  }

  override def project(v: BSV[Double], metadata: Metadata): BSV[Double] = {
    if (inequality) {
      val intermediate = unitBoxProjection.project(v, metadata)
      if (intermediate.data.sum <= 1) {
        return intermediate
      }
    }
    equalityConstraint(v)
  }

  override def maxNorm(size: Int): Double = {
    1.0
  }

  override def minNorm(size: Int): Double = {
    if (inequality) {
      0.0
    } else {
      1.0 / size
    }
  }

  override def isVertexSolution(v: BSV[Double], metadata: Metadata): Boolean = {
    if (inequality) {
      if (v.activeSize <= 1 && v.data.forall(x => x == 1.0)) {
        return true
      }
    } else {
      if (v.activeSize == 1 && v.data.forall(x => x == 1.0)) {
        return true
      }
    }
    false
  }
}