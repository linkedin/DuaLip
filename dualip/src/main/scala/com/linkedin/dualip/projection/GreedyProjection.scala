package com.linkedin.dualip.projection

import breeze.linalg.{argmax, SparseVector => BSV}


/**
  * Greedy inequality projection. Let j = argmax_i x_i.
  * - If x_j >= 0, return the vector with 1 in the jth position, 0 otherwise.
  * - If x_j < 0, return a zero vector.
  */
class GreedyProjection() extends Projection with Serializable {

  /**
    * Arg max operation to pick the best vertex.
    *
    * @param v    Input vector v.
    * @return     The projected vector.
    */
  override def project(v: BSV[Double], metadata: Metadata): BSV[Double] = {
    val index = argmax(v)
    if (v(index) > 0) {
      new BSV(Array(index), Array(1.0), v.size)
    } else {
      new BSV(Array(), Array(), v.size)
    }
  }

  override def maxNorm(size: Int): Double = ???
  
  override def minNorm(size: Int): Double = ???
}