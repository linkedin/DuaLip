package com.linkedin.dualip.projection

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV}
import breeze.optimize.proximal.Constraint.{BOX, Constraint, POSITIVE, PROBABILITYSIMPLEX}
import breeze.optimize.proximal.QuadraticMinimizer

/**
  * Every linear system of squares can be written as 0.5 ‖𝑄𝑥 − 𝑐‖^2  which is
  * 0.5(𝑄𝑥−𝑐)^𝑇(𝑄𝑥−𝑐) = 0.5(𝑥^𝑇𝑄^𝑇𝑄𝑥 − 𝑥^𝑇𝑄^𝑇𝑐 − 𝑐^𝑇𝑄𝑥 + 𝑐^𝑇𝑐).
  * However, since c^Tc is a fixed quantity, it is sufficient to solve the Quadratic Programming problem:
  *   f(𝑥)=0.5 𝑥^𝑇𝐴𝑥 +𝑞^𝑇𝑥 where 𝐴=𝑄^𝑇𝑄 and 𝑞 =−𝑄^𝑇𝑐
  *
  * This solver is used to solve f(𝑥) + g(𝑥) where g(𝑥) covers one of the following Constraint settings:
  *  SMOOTH: L2 regularization with lambda
  *  SPARSE: L1 regularization with lambda
  *  POSITIVE: x >= 0
  *  BOX: 0 <= x <= 1
  *  PROBABILITYSIMPLEX: 1^Tx = lambda
  */
class QPProjection(constraint: Constraint = PROBABILITYSIMPLEX, lambda: Double = 1.0) extends Projection with Serializable {
  override def project(v: BSV[Double], metadata: Metadata): BSV[Double] = {
    val A = BDM.eye[Double](v.size)
    val q = BDV.apply(v.data)
    val qpSolver = QuadraticMinimizer(v.size, constraint, lambda)
    val projection = qpSolver.minimize(A, q *:* (-1.0))
    val result = BSV.apply(projection.data)
    result.compact()
    result
  }

  override def maxNorm(size: Int): Double = {
    constraint match {
      case PROBABILITYSIMPLEX => 1.0
      case BOX => 1.0
      case POSITIVE => Double.PositiveInfinity
      case _ => ???
    }
  }
  override def minNorm(size: Int): Double = {
    constraint match {
      case PROBABILITYSIMPLEX => 1.0 / size
      case BOX => 0.0
      case POSITIVE => 0.0
      case _ => ???
    }
  }
}