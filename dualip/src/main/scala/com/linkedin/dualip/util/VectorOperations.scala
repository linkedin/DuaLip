package com.linkedin.optimization.util

import breeze.linalg.{SparseVector => BSV}

/**
  * A collection of distributed/local vector operations
  */
object VectorOperations {

  /** Count the number of nonzeros in a vector.
    *
    * @param v Input vector v
    * @return The number of non-zeros in a vector upto tolerance
    */
  def countNonZeros(v: BSV[Double], eps: Double = 1e-6): Int = {
    var nnz = 0
    v.data.foreach { v =>
      if (Math.abs(v) > eps) {
        nnz += 1
      }
    }
    nnz
  }

  /**
    * Dot product of two sparse vectors.
    * Note: creating this function due to a potential bug in breeze dot product implementation
    */
  def dot(x: BSV[Double], y: BSV[Double]): Double = {
    val xValues = x.data
    val xIndices = x.index
    val yValues = y.data
    val yIndices = y.index
    val nnzx = xIndices.length
    val nnzy = yIndices.length

    var kx = 0
    var ky = 0
    var sum = 0.0
    // y catching x
    while (kx < nnzx && ky < nnzy) {
      val ix = xIndices(kx)
      while (ky < nnzy && yIndices(ky) < ix) {
        ky += 1
      }
      if (ky < nnzy && yIndices(ky) == ix) {
        sum += xValues(kx) * yValues(ky)
        ky += 1
      }
      kx += 1
    }
    sum
  }

  /**
    * Element-wise multiplicaton of sparse vector by a scalar
    *
    * @param x sparse vector
    * @param s scalar double
    * @return a sparse vector post multiplication
    */
  def multiply(x: BSV[Double], s: Double): BSV[Double] = {
    val values = x.data.map(_ * s)
    new BSV(x.index, values, x.size)
  }

  /**
    * Element-wise multiplicaton of an array by a scalar
    *
    * @param x array of doubles
    * @param s scala double
    * @return a array post multiplication
    */
  def multiply(x: Array[Double], s: Double): Array[Double] = {
    x.map(_ * s)
  }

  /**
    * Element-wise sum of a list of sparse vectors.
    *
    * @param xs list of sparse vectors
    * @return
    */
  def sum(xs: Seq[BSV[Double]]): BSV[Double] = {
    xs.reduce(_ + _)
  }

  /**
    * Element-wise sum of two arrays
    *
    * @param x the first array
    * @param y the second array
    * @return
    */
  def sum(x: Array[Double], y: Array[Double]): Array[Double] = {
    (x, y).zipped.map(_ + _)
  }

  /**
    * Utility method
    *
    * @param data The dataset to be converted
    * @param size The size of the data
    * @return
    */
  def toBSV(data: Array[(Int, Double)], size: Int): BSV[Double] = {
    val (indices, values) = data.sortBy { case (index, _) => index }.unzip
    new BSV(indices, values, size)
  }
}
