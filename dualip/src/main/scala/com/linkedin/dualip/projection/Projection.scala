package com.linkedin.dualip.projection

import breeze.linalg.{SparseVector => BSV}


/*
 *  This file defines major APIs for projecting the primal variable.
 */

/**
  * Generic API of the projection operator. It takes the unconstrained variable as the input and projects it to
  * honor the "simple" constraints in the problem definition.
  */
trait Projection {

  type Metadata = Map[String, Double]
  /**
    * @param v Unconstrained primal variable.
    * @return Projected primal variable.
    */
  def project(v: BSV[Double], metadata: Metadata): BSV[Double]

  /**
    * The max norm (`x^Tx`) possible using the defined projection.
    * @param size Size of the variable.
   * @return
   */
  def maxNorm(size: Int): Double = ???

  /**
    * The min norm (`x^Tx`) possible using the defined projection.
    * @param size Size of the variable.
   * @return
   */
  def minNorm(size: Int): Double = ???

  /**
    * To check if the projection returns a vertex solution.
    * @param v - Projected primal variable.
   * @return
   */
  def isVertexSolution(v: BSV[Double], metadata: Metadata): Boolean = ???
}