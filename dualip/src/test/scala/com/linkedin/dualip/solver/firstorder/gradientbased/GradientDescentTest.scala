package com.linkedin.dualip.solver.firstorder.gradientbased

import breeze.linalg.{SparseVector => BSV}
import org.testng.Assert
import org.testng.annotations.Test

class GradientDescentTest {

  @Test
  def testSimpleSolverConstrained(): Unit = {
    val solver = new GradientDescent(maxIter = 30)
    val (solution, _, _) = solver.maximize(new SimpleObjective(), BSV(Array(0.0, 0.0)))
    val Array(x, y) = solution.toArray
    // Unconstrained solution in this case is (3, -5) but it should not find it
    Assert.assertTrue(Math.abs(x - 3.0) < 1e-1)
    Assert.assertTrue(Math.abs(y - 0.0) < 1e-1)
  }
}