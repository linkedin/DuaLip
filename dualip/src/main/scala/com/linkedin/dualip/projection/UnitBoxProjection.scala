package com.linkedin.dualip.projection

import breeze.linalg.{SparseVector => BSV}


/**
  * UnitBox projection, 0 <= x_j <= 1
  *
  */
class UnitBoxProjection() extends Projection with Serializable {

  /**
    * Set negative values to 0, set values larger than 1 to 1.
    *
    * @param v    Input vector v
    * @return     The projected vector
    */
  override def project(v: BSV[Double], metadata: Metadata): BSV[Double] = {
    val values = v.data.map(x =>
      if (x > 1) 1.0
      else if (x < 0) 0.0
      else x)

    val result = new BSV(v.index, values, v.size)
    // compact() will have all explicit zeros removed.
    result.compact()
    result
  }

  override def maxNorm(size: Int): Double = 1.0
  override def minNorm(size: Int): Double = 0.0

  override def isVertexSolution(v: BSV[Double], metadata: Metadata): Boolean = {
    v.activeSize == 0 || v.data.forall(x => x == 1.0)
  }
}